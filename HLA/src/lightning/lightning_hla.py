from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from src.hla.matcher import HLA
from src.hla.backbone import ConvNeXtV2Backbone
from src.hla.utils.supervision import compute_supervision_fine, compute_supervision_coarse
from src.losses.loss import Loss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_f1,
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics_train_val, aggregate_metrics_test
)
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from src.utils.plotting import make_matching_figures
from src.utils.review_diagnostics import (
    ReviewDiagnostics, ReviewFlopCounter, collect_review_flops,
)
from src.utils.pfm_training_diagnostics import PFMTrainingDiagnostics


class PL_HLA(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.HLA_cfg = lower_config(_config['hla'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)
        self.viz_path = Path('outputs/visualization')
        self.viz_path.mkdir(parents=True, exist_ok=True)
        # Matcher: HLA
        self.backbone = ConvNeXtV2Backbone()
        self.matcher = HLA(config=_config['hla'])
        self.loss = Loss(_config)

        if pretrained_ckpt:
            from src.hla.checkpoints import load_hla_state_dict
            state_dict = load_hla_state_dict(pretrained_ckpt)
            self.load_state_dict(state_dict, strict=True)
            logger.info('Loaded HLA checkpoint from {}', pretrained_ckpt)

        # Training diagnostics are independent of the inference timing/FLOPs pass.
        self.pfm_training_diagnostics = PFMTrainingDiagnostics()
        for layer_index, (block, kind) in enumerate(zip(
                self.matcher.hla_interaction.atten.layers,
                self.matcher.hla_interaction.atten.layer_names)):
            block.attn.pfm_label = f"layer{layer_index}_{kind}_order{block.attn.order}"
            if self.pfm_training_diagnostics.enabled:
                block.attn.pfm_diagnostics = self.pfm_training_diagnostics
        self.pfm_mode = self.matcher.hla_interaction.atten.layers[0].attn.pfm_mode

        # Testing
        self.dump_dir = dump_dir
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0
        self.total_test_pairs = 0
        self.runtime_per_pair_ms = []
        self.review_diagnostics = ReviewDiagnostics()
        active_timer = self.review_diagnostics if self.review_diagnostics.timing_enabled else None
        self.matcher.review_timer = active_timer
        self.matcher.hla_interaction.review_timer = active_timer
        self.review_flop_counter = None
        self.total_parameter_count = sum(p.numel() for p in self.parameters())
        if self.review_diagnostics.flops_enabled:
            self.review_flop_counter = ReviewFlopCounter(self.review_diagnostics.output_dir)
            self.review_flop_counter.attach(self.backbone, "backbone")
            self.review_flop_counter.attach(self.matcher, "matcher")
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('number of params:', n_parameters / 1e6)

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                     (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                     abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _train_inference(self, batch):

        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("Backbone"):
            with torch.autocast(enabled=self.config.HLA.MP, device_type='cuda'):
                self.backbone(batch)

        with self.profiler.profile("Matcher"):
            self.matcher(batch, mode='train')

        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=self.config.HLA.MP, device_type='cuda'):
                compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.HLA.MP, device_type='cuda'):
                self.loss(batch)

    def _val_inference(self, batch):

        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("Backbone"):
            with torch.autocast(enabled=self.config.HLA.MP, device_type='cuda'):
                self.backbone(batch)

        with self.profiler.profile("Matcher"):
            self.matcher(batch, mode='val')

        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=self.config.HLA.MP, device_type='cuda'):
                compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.HLA.MP, device_type='cuda'):
                self.loss(batch)

    def _compute_metrics_val(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_f1(batch)  # compute fi-score
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['imagec_0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'precision': batch['precision'],
                'recall': batch['recall'],
                'f1_score': batch['f1_score'],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['imagec_0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):
        self.pfm_training_diagnostics.start_step(
            int(self.global_step), batch_idx, self.pfm_mode, bool(self.config.HLA.MP)
        )
        try:
            self._train_inference(batch)
            self.pfm_training_diagnostics.forward_complete(batch)
        except Exception as exc:
            self.pfm_training_diagnostics.exception(batch, exc)
            raise
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            self.logger.experiment.add_scalar(f'train_loss', batch['loss'], self.global_step)

        return {'loss': batch['loss']}

    def on_after_backward(self):
        # In the default FP32 configuration these are unscaled gradients. When
        # AMP is enabled, this hook may see scaled gradients; the log marks it.
        self.pfm_training_diagnostics.after_backward(self)

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        self._val_inference(batch)
        ret_dict, _ = self._compute_metrics_val(batch)
        ret_dict['metrics'] = {**ret_dict['metrics'], 'max_matches': [batch['num_candidates_max']]}
        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}

        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }

    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if self.trainer.sanity_checking:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics_train_val(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])

            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this
        print(val_metrics_4tb)

    def test_step(self, batch, batch_idx):
        pair_count = int(batch['imagec_0'].shape[0])
        review_metadata = {
            'input_height': int(batch['imagec_0'].shape[-2]),
            'input_width': int(batch['imagec_0'].shape[-1]),
            'device': str(batch['imagec_0'].device),
            'image_dtype': str(batch['imagec_0'].dtype),
            'mixed_precision': bool(self.config.HLA.MP),
        }
        self.review_diagnostics.start_batch(batch_idx, pair_count, metadata=review_metadata)
        if self.review_flop_counter is not None:
            self.review_flop_counter.start_batch(batch_idx, pair_count)
            flop_scope = collect_review_flops(self.review_flop_counter)
        else:
            from contextlib import nullcontext
            flop_scope = nullcontext()
        with torch.autocast(enabled=self.config.HLA.MP, device_type='cuda'):
            self.start_event.record()
            with flop_scope:
                with self.review_diagnostics.measure("model_total"):
                    with self.profiler.profile("Backbone"):
                        with self.review_diagnostics.measure("backbone"):
                            self.backbone(batch)

                    with self.profiler.profile("Matcher"):
                        with self.review_diagnostics.measure("matcher_total"):
                            self.matcher(batch, mode='test')

            self.end_event.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
            batch['runtime'] = self.start_event.elapsed_time(self.end_event)
            self.total_test_pairs += pair_count
            self.runtime_per_pair_ms.append(batch['runtime'] / max(pair_count, 1))
            self.review_diagnostics.finish_batch(synchronized=True)
            if self.review_flop_counter is not None:
                self.review_flop_counter.finish_batch(
                    input_height=batch['imagec_0'].shape[-2],
                    input_width=batch['imagec_0'].shape[-1],
                    parameter_count=self.total_parameter_count,
                    metadata=review_metadata,
                )

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        # Visualization #
        path = str(self.viz_path) + '/' + str(batch_idx)
        # make_matching_figures(batch, 'confidence', path=path+'_confidence.png')
        # make_matching_figures(batch, 'evaluation', path=path+'_evaluation.png')
        # make_matching_figures(batch, 'wheel', path=path+'_wheel.png')

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                bs = batch['imagec_0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    epi_errs = batch['epi_errs'][mask].cpu().numpy()
                    correct_mask = epi_errs < 1e-4
                    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
                    n_correct = np.sum(correct_mask)
                    item['precision'] = precision
                    item['n_correct'] = n_correct
                    item['runtime'] = batch['runtime']
                    for key in ['R_errs', 't_errs']:
                        item[key] = batch[key][b_id][0]
                    dumps.append(item)
                ret_dict['dumps'] = dumps
        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics_test(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            mean_ms = self.total_ms / max(self.total_test_pairs, 1)
            runtime_tensor = torch.tensor(self.runtime_per_pair_ms, dtype=torch.float64)
            if runtime_tensor.numel():
                median_ms = float(runtime_tensor.median().item())
                p90_ms = float(torch.quantile(runtime_tensor, 0.90).item())
            else:
                median_ms = p90_ms = 0.0
            print(
                'Matching network time over {} pairs: mean {:.2f} ms/pair, '
                'median {:.2f} ms/pair, p90 {:.2f} ms/pair'.format(
                    self.total_test_pairs, mean_ms, median_ms, p90_ms
                )
            )
            if self.review_diagnostics.timing_enabled:
                summary = self.review_diagnostics.summary()
                logger.info('Review stage timing summary (ms/pair): {}', summary['stages'])
                for name, stats in summary['stages'].items():
                    print(
                        '  {:48s} mean={:.3f} median={:.3f} p90={:.3f} ms/pair'.format(
                            name, stats['mean_ms_per_pair'],
                            stats['median_ms_per_pair'], stats['p90_ms_per_pair']
                        )
                    )
            if self.review_flop_counter is not None:
                flops_summary = self.review_flop_counter.summary()
                print(
                    'Estimated FLOPs: {:.4f} GFLOPs/pair; stage mean: {}'.format(
                        flops_summary['mean_gflops_per_pair'],
                        flops_summary['mean_gflops_per_pair_by_stage']
                    )
                )
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'HLA_pred_eval', dumps)
        #    # === 新增：计算并保存 0-20度 CEC 曲线数据 ===
        #     # 1. 提取所有图像对的 R_err 和 t_err
        #     R_errs = np.array(metrics['R_errs'])
        #     t_errs = np.array(metrics['t_errs'])
            
        #     # 2. 位姿误差取 R 和 t 误差的最大值 (标准的 MegaDepth 评估方式)
        #     pose_errs = np.maximum(R_errs, t_errs)
            
        #     # 3. 生成 0 到 20 度的连续阈值（比如划分 200 个点，步长 0.1 度）
        #     thresholds = np.linspace(0, 20, 200)
        #     cumulative_accuracy = []
            
        #     # 4. 计算每个阈值下的精度
        #     for thr in thresholds:
        #         acc = np.mean(pose_errs <= thr) * 100  # 乘以 100 转换为百分比
        #         cumulative_accuracy.append(acc)
                
        #     # 5. 保存数据到文件，用于后续画图
        #     save_path = Path(self.dump_dir) if self.dump_dir else Path('.')
        #     save_path.mkdir(parents=True, exist_ok=True)
        #     np.save(save_path / 'cec_curve_ours.npy', {
        #         'thresholds': thresholds, 
        #         'accuracy': cumulative_accuracy
        #     })
        #     logger.info(f'🎉 CEC curve data [0°-20°] saved to: {save_path / "cec_curve_ours.npy"}')
        #     # =========================================================

        #     if self.dump_dir is not None:
        #         np.save(Path(self.dump_dir) / 'HLA_pred_eval', dumps)
    # def test_step(self, batch, batch_idx):
    #     flops1, params1 = profile(self.backbone, inputs=[batch])
    #     flops2, params2 = profile(self.matcher, inputs=[batch])
    #     return flops1+flops2
    
    # def test_epoch_end(self, outputs):
    #     flops_mean = sum(outputs)/len(outputs) / 1e9
    #     print("mean flops: {}G".format(flops_mean))


