# HLA method-to-code mapping

| Paper component | Code implementation |
|---|---|
| Paired channels as complex values | `src/hla/harmonic_attention.py`, `HarmonicLinearAttention.forward/get_phi` |
| Amplitude `r=abs(z)` and magnitude factor `Softplus(r)` | `get_phi` |
| Normalize complex channels with epsilon, raise to harmonic order | `get_phi` |
| Four-branch signed PFM | `PreservationFeatureMap.forward` |
| First-order self-attention and third-order cross-attention | `HLAInteractionStack.__init__` |
| Factorized `Phi(Q) @ (Phi(K)^T @ V)` | `HarmonicLinearAttention.forward` |
| Local enhanced positional encoding (LEPE) | depthwise convolution `lepe` in `HarmonicLinearAttention` |
| Coarse matching via two directional softmaxes and union | `src/hla/matching_module.py`, `CoarseMatching` |
| Fine-level dual-softmax, MNN and subpixel offsets | `src/hla/matching_module.py`, `FineSubMatching` |
| Loss terms | `src/losses/loss.py` and `src/hla/utils/supervision.py` |

The implementation's default PFM is non-negative, but special ablation modes
may not be. Current image-pair matching code expects the two images to be
resized/padded to equal coarse-grid resolution.

Important numerical note: the original attention applies token masks to Q/K/V
*before* the Softplus feature map. A zero masked vector becomes positive after
Softplus, so a masked token can still contribute to PFM kernel mass. We did not
silently change this default masking order because that would affect the
model's learned behavior and require fresh benchmarking; verify if you use
variable-padding masks and document any changes in subsequent experiments.
