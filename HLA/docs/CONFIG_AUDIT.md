# Training hyperparameter audit: confirm before paper-reproduction claims

The uploaded source configuration was retained **without changing numerical
training settings**, so the public refactor does not silently alter the run.

| Source of information | Learning rate | Linear warm-up |
|---|---|---|
| Manuscript implementation section | `2e-4` | 1 epoch |
| Uploaded `configs/hla/outdoor/final.py` | `CANONICAL_LR=1e-4` | `WARMUP_STEP=58200` (comment: 3 epochs) |

In `train.py`, the effective learning rate is `CANONICAL_LR * (WORLD_SIZE *
batch_size / CANONICAL_BS)` and warm-up steps are divided by this same
scaling. For the provided one-GPU batch-size-2 command and
`CANONICAL_BS=2`, the scaling is 1.0, so the code requests learning rate
`1e-4` and warm-up `58200` optimizer steps.

This discrepancy cannot be resolved merely by changing words or variables.
Check the **training checkpoint/log/config that generated the reported
78.1% MegaDepth AUC@10°**. If the manuscript is correct, update the training
config to the actual values. If the source config is correct, update the
manuscript implementation details. Do not report the released configuration
as an exact reproduction until this has been reconciled.
