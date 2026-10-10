import torch
from torch import nn
import torch.nn.functional as F
from einops.einops import rearrange
from contextlib import nullcontext
from src.hla.utils.utils import KeypointEncoder_wo_score, up_conv4, MLPMixerEncoderLayer, normalize_keypoints
from src.hla.harmonic_attention import HLAFeatureInteraction
from src.hla.matching_module import CoarseMatching, FineSubMatching
from src.utils.profiler import PassThroughProfiler
from src.utils.review_diagnostics import add_review_flops
import pdb
torch.backends.cudnn.deterministic = True
INF = 1E9


class HLA(nn.Module):
    def __init__(self, config, profiler=None):
        super().__init__()
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        # Set by PL_HLA only for opt-in review timing.
        self.review_timer = None
        self.d_model_c = self.config['coarse']['d_model']
        self.d_model_f = self.config['fine']['d_model']

        self.kenc = KeypointEncoder_wo_score(self.d_model_c, [32, 64, 128, self.d_model_c])
        self.hla_interaction = HLAFeatureInteraction(self.d_model_c)
        self.coarse_matching = CoarseMatching(config['match_coarse'], self.profiler)

        self.act = nn.GELU()
        dim = [256, 128, 64]
        self.up2 = up_conv4(dim[0], dim[1], dim[1])  # 1/8 -> 1/4
        self.conv7a = nn.Conv2d(2*dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.conv7b = nn.Conv2d(dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.up3 = up_conv4(dim[1], dim[2], dim[2])  # 1/4 -> 1/2
        self.conv8a = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)
        self.conv8b = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)

        W = self.config['fine_window_size']
        self.fine_enc = nn.ModuleList([MLPMixerEncoderLayer(2*W**2, 64) for _ in range(4)])
        self.fine_matching = FineSubMatching(config, self.profiler)

    def coarse_match(self, data):
        timer = self.review_timer
        measure = timer.measure if timer is not None else lambda _name: nullcontext()
        with measure("coarse_preprocess"):
            desc0, desc1 = data['feat_8_0'].flatten(2, 3), data['feat_8_1'].flatten(2, 3)

            kpts0, kpts1 = data['grid_8'], data['grid_8']
            # Keypoint normalization.
            kpts0 = normalize_keypoints(kpts0, data['imagec_0'].shape[-2:])
            kpts1 = normalize_keypoints(kpts1, data['imagec_1'].shape[-2:])

            kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
            desc0 = desc0 + self.kenc(kpts0)
            desc1 = desc1 + self.kenc(kpts1)
            add_review_flops("coarse_preprocess.keypoint_feature_add", desc0.numel() + desc1.numel())
            data.update({
                'feat_8_0': desc0,
                'feat_8_1': desc1,
            })

        with self.profiler.profile("coarse interaction"):
            with measure("hla_interaction"):
                self.hla_interaction(data)

        # 3. match coarse-level
        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        with self.profiler.profile("coarse matching"):
            with measure("coarse_matching"):
                self.coarse_matching(data['feat_8_0'], data['feat_8_1'], data, mask_c0=mask_c0, mask_c1=mask_c1)

    # def inter_fpn(self, feat_8, feat_4):
    #     d2 = self.up2(feat_8)  # 1/4
    #     d2 = self.act(self.conv7a(torch.cat([feat_4, d2], 1)))
    #     feat_4 = self.act(self.conv7b(d2))

    #     d1 = self.up3(feat_4)  # 1/2
    #     d1 = self.act(self.conv8a(d1))
    #     feat_2 = self.conv8b(d1)
    #     return feat_2
    def inter_fpn(self, feat_8, feat_4):
        # === 原始代码 ===
        d2 = self.up2(feat_8)  # 1/4
        
        # === 关键修复：强制对齐空间尺寸 ===
        # 检查并修正尺寸差异（容忍1像素误差）
        if d2.shape[2] != feat_4.shape[2] or d2.shape[3] != feat_4.shape[3]:
            d2 = F.interpolate(
                d2,
                size=(feat_4.shape[2], feat_4.shape[3]),  # 严格匹配feat_4的尺寸
                mode='bilinear',
                align_corners=False,  # PyTorch官方推荐设置
                antialias=True        # 减少插值伪影（PyTorch 1.9+）
            )
            # Approximate bilinear interpolation arithmetic for the actual output.
            # Any extra work from antialias filtering is not included in this estimate.
            add_review_flops("fine_matching.functional_bilinear_alignment", 7 * d2.numel())
        
        # === 继续原始逻辑 ===
        d2 = self.act(self.conv7a(torch.cat([feat_4, d2], 1)))
        feat_4 = self.act(self.conv7b(d2))

        d1 = self.up3(feat_4)  # 1/2
        d1 = self.act(self.conv8a(d1))
        feat_2 = self.conv8b(d1)
        return feat_2
    def fine_preprocess(self, data, profiler):
        
        data['resolution1'] = 8
        stride = data['resolution1'] // self.config['resolution'][1]
        W = self.config['fine_window_size']
        feat_8 = torch.cat([data['feat_8_0'], data['feat_8_1']], 0).view(2*data['bs'], data['c'], data['h_8'], -1)
        feat_4 = torch.cat([data['feat_4_0'], data['feat_4_1']], 0)
        # feat_2 = torch.cat([data['feat_2_0'], data['feat_2_1']], 0)
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, W ** 2, self.d_model_f, device=feat_4.device)
            feat1 = torch.empty(0, W ** 2, self.d_model_f, device=feat_4.device)
            return feat0, feat1

        # feat_f = self.inter_fpn(feat_8, feat_4, feat_2)
        feat_f = self.inter_fpn(feat_8, feat_4)
        feat_f0, feat_f1 = torch.chunk(feat_f, 2, dim=0)
        
        data.update({'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]})

        # 1. unfold(crop) all local windows
        pad = 0 if W % 2 == 0 else W//2
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=pad)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=pad)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)  # [b, h_f/stride * w_f/stride, w*w, c]

        # 2. select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]  # [n, ww, cf]
        
        feat_f = torch.cat([feat_f0_unfold, feat_f1_unfold], 1).transpose(1, 2)
        for layer in self.fine_enc:
            feat_f = layer(feat_f)
        feat_f0_unfold, feat_f1_unfold = feat_f[:, :, :W**2], feat_f[:, :, W**2:]
        return feat_f0_unfold, feat_f1_unfold

    def forward(self, data, mode='test'):
        self.mode = mode
        data.update({
            'hw0_i': data['imagec_0'].shape[2:],
            'hw1_i': data['imagec_1'].shape[2:],
            'hw0_c': [data['h_8'], data['w_8']],
            'hw1_c': [data['h_8'], data['w_8']],
        })

        self.coarse_match(data)

        with self.profiler.profile("fine matching"):
            # 4. fine-level matching module
            timer = self.review_timer
            measure = timer.measure if timer is not None else lambda _name: nullcontext()
            with measure("fine_matching"):
                with measure("fine_preprocess"):
                    feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(data, self.profiler)

                # 5. match fine-level and sub-pixel refinement
                with measure("fine_submatching"):
                    self.fine_matching(feat_f0_unfold.transpose(1, 2), feat_f1_unfold.transpose(1, 2), data)
