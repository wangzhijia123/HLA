# Train HLA on MegaDepth

## Pre-flight checks

- Linux, NVIDIA GPU and compatible PyTorch/CUDA build.
- MegaDepth/D2-Net dataset and evaluation indices in the paths specified by
  `configs/data/megadepth_trainval_832.py` (see [DATASETS.md](DATASETS.md)).
- A compatible pretrained **ConvNeXt V2 Nano** backbone checkpoint. It is
  required to match the initialization used in the paper; HLA does not
  automatically download weights from another project.

## Example

```bash
export HLA_BACKBONE_WEIGHTS=/absolute/path/to/convnextv2_nano_pretrain.ckpt
bash scripts/train_megadepth.sh
```

Default HLA configuration uses 256-channel coarse features, 64-channel fine
features, first-order harmonic self-attention, third-order harmonic
cross-attention, focal-based matching losses and an epipolar refinement loss.
The script requests 30 epochs and batch size 2 on a GPU. The training command
assumes the provided configuration paths are valid.

**Before claiming exact paper reproduction, read [CONFIG_AUDIT.md](CONFIG_AUDIT.md).**
The uploaded config does not exactly match the manuscript's stated learning
rate and warm-up. Both settings have been preserved for auditability, not
changed to an unverified guess.

The Lightning checkpoints/logs are written under `outputs/tensorboard/`.
Keep trained weights private until you have verified the right to release them.

**Important:** There are no included pre-trained HLA model weights or results
that can independently verify the paper's reported AUCs. Do not compare random
initialization to the reported experimental results.
