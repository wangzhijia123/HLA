# Before announcing reproduction of the paper's results

- [ ] Install all dependencies and run `python -m pytest tests -q`.
- [ ] Supply a complete HLA checkpoint trained for the final model architecture.
- [ ] Run the full MegaDepth 1500-pair test and verify AUC@5/10/20.
- [ ] Run HPatches evaluation with the same homography/RANSAC protocol as the paper.
- [ ] Run ScanNet with the MegaDepth-trained checkpoint and disclose domain shift.
- [ ] Resolve the specific learning-rate and warm-up discrepancy in docs/CONFIG_AUDIT.md against the training run logs.
- [ ] Check exact training split, optimizer schedule, hardware and output resolutions.
- [ ] Make any usable pretrained ConvNeXt V2 Nano weights accessible with their license.
- [ ] Obtain appropriate rights for included data/index metadata.
- [ ] Retain LICENSE and THIRD_PARTY_NOTICES.md in the public repository.
- [ ] Remove personal paths, API keys, and any checkpoints not approved for publication.

Passing unit tests validates formula shapes/basic numerical behavior only; it
is NOT evidence that the reported 78.1% pose AUC has been reproduced.
