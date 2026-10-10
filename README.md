# HLA: Harmonic Linear Attention for Compact Semi-Dense Local Feature Matching

Official research-code release for **HLA**, a compact local feature matching method based on complex-domain harmonic linear attention.

**Authors:** Zhijia Wang, Yong Wang, Pei Wang (Jilin University)  
**Manuscript:** *Harmonic Linear Attention for Compact Semi-Dense Local Feature Matching*  
**Project:** https://github.com/wangzhijia123/HLA

> **Release status:** Source code and evaluation metadata are included. **The trained
> 4.8M-parameter checkpoint is not part of this archive.** The MegaDepth AUC@10°
> of 78.1% is a result reported in the manuscript, **not a result verified by this
> packaging process**. You must supply your own corresponding trained checkpoint
> to run meaningful inference or reproduce the reported benchmark.

## Main components

- Harmonic complex-valued representation of paired feature channels.
- Magnitude-related weighting and first-/third-order phase modulation.
- Four-branch Preservation Feature Map (PFM): positive/negative real and imaginary responses.
- Factorized global linear attention with local enhancement, followed by coarse-to-fine matching.
- Self-attention: order **1**; cross-attention: order **3** (8 attention heads).

## Repository layout

```text
src/hla/                HLA model, harmonic attention, backbone and matching
src/lightning/          training/evaluation Lightning wrappers
configs/hla/            indoor/outdoor model configurations
configs/data/           benchmark data configurations
demo/                   image-pair inference (requires weights)
scripts/                example training and evaluation commands
tests/                  CPU attention tests and structural checks
docs/                   dataset, weights, reproducibility and attribution notes
assets/                 small released benchmark evaluation metadata
```

## Installation

Use Linux and a compatible NVIDIA GPU for the full training and evaluation
pipeline. The included attention unit tests also run on CPU.

```bash
conda create -n hla python=3.10 -y
conda activate hla
# Install PyTorch + torchvision for your CUDA toolkit, following pytorch.org.
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The original training code uses the **PyTorch Lightning 1.x** trainer interface.
Other dependency combinations may require adaptations; the full GPU pipeline
has **not** been executed during this release refactor.

## Use pretrained HLA weights for inference

A **full HLA Lightning checkpoint** must be supplied via `HLA_CKPT` or `--ckpt_path`.
No checkpoint is silently downloaded, and a weight file from a different matcher
is **not** compatible with this HLA architecture.

```bash
export HLA_CKPT=/absolute/path/to/your/hla_model.ckpt
bash scripts/eval_megadepth.sh
bash scripts/eval_scannet.sh  # zero-shot HLA: MegaDepth checkpoint on ScanNet
```

For your own image pair:

```bash
python -m demo.demo --image1 /path/to/a.jpg --image2 /path/to/b.jpg \
  --ckpt_path "$HLA_CKPT" --output_dir outputs/demo
```

For dataset paths and training commands see [docs/DATASETS.md](docs/DATASETS.md)
and [docs/TRAINING.md](docs/TRAINING.md). For checkpoint naming and compatibility,
see [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).

## Training

To reproduce the paper's initialization you must provide the compatible
pretrained ConvNeXt V2 Nano backbone weights (not included):

```bash
export HLA_BACKBONE_WEIGHTS=/absolute/path/to/convnextv2_nano_pretrain.ckpt
bash scripts/train_megadepth.sh
```

Training from randomly initialized backbone weights is **not** equivalent
to the setup used for the reported paper results.

**Training configuration verification required:** The uploaded experimental
configuration specifies `CANONICAL_LR=1e-4`, `CANONICAL_BS=2` and
`WARMUP_STEP=58200` (annotated as 3 epochs). The manuscript states initial
learning rate `2e-4` and a 1-epoch linear warm-up. We intentionally retained
the uploaded values to avoid changing the experiment without knowing which
settings produced the reported checkpoint. See [docs/CONFIG_AUDIT.md](docs/CONFIG_AUDIT.md).

## Benchmark claims

- **MegaDepth:** 78.1% pose AUC@10°; model size 4.8M parameters (reported in paper).
- **HPatches:** Homography evaluation described in paper; benchmark runner is not included here.
- **ScanNet:** MegaDepth-trained HLA tested without indoor fine-tuning. Baseline
  weights may have different training histories; their numbers do not constitute
  a matched zero-shot baseline comparison.

## Attribution, license and citation

HLA's attention mechanism is the research contribution of this project.
The feature extraction/matching/training scaffold incorporates third-party
components. Source notices and original copyright are preserved in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE).

If citing the HLA manuscript, see [CITATION.cff](CITATION.cff). Please also cite
relevant baseline/data papers when using the provided benchmark setup.
