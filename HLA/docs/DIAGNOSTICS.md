# HLA diagnostics

The original experiment code retains optional hooks for timing, FLOPs, and
PFM-related ablations; these are **not** enabled in the standard benchmark.

- `HLA_PFM_MODE=full` (default): four positive Softplus branches.
- `HLA_PFM_MODE=phase_only` and `amplitude4`: component ablations.
- `HLA_PFM_MODE=raw_ri`, `signed4`, `relu4`, `elu4`: experimental alternatives.

The default attention remains the configuration used for the paper results.
Caution: some ablation maps are signed and do not satisfy the non-negative
kernel argument of the main model.
