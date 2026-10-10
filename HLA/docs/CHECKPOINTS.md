# Checkpoint compatibility

The refactor changed public Python import paths, class names and one internal
module name (`matcher.hla_interaction`). The underlying module tensor names
inside the attention stack have otherwise been kept stable.

`src/hla/checkpoints.py` translates the older local checkpoint key
`matcher.joint_mamba.` into `matcher.hla_interaction.` automatically. The loader
accepts a full PyTorch Lightning checkpoint containing `state_dict`, or a raw
model state_dict. Load **only trusted checkpoints**, since unpickling checkpoint
metadata may execute arbitrary code.

The backbone's original pretrained initialization is **not bundled**. Training
requires setting `HLA_BACKBONE_WEIGHTS` to compatible parameters before running
`train.py`. During evaluation, a complete HLA checkpoint overrides model weights
including the backbone.

No attempt is made to translate weights from other model architectures into HLA.
Validate model parameter counts and benchmark scores after uploading your weights.
