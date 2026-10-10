# Third-party origin and licenses

This repository is presented as the **HLA research project**. Its harmonic
attention and PFM are the focus of the manuscript, while parts of the backbone,
coarse-to-fine matcher, Lightning training/testing scaffold and metrics are
adapted from previously published open-source projects.

The initial project scaffold was derived from **JamMa** (Lu and Du, CVPR 2025),
https://github.com/leoluxxx/JamMa, provided under the MIT License
(Copyright (c) 2025 leo). The copyright and license notice are retained in
`LICENSE`. Renaming namespaces and entry points does **not** transfer ownership
of those upstream components.

The original scaffold acknowledges code from:

- **LoFTR** (Sun et al., CVPR 2021): https://github.com/zju3dv/LoFTR
  (Apache License 2.0; a copy is provided in `licenses/APACHE-2.0.txt`).
- **XoFTR** (Tuzcuoglu et al., CVPRW 2024): https://github.com/OnderT/XoFTR
  (Apache License 2.0; a copy is provided in `licenses/APACHE-2.0.txt`).
- **ConvNeXt V2** backbone implementation (Woo et al., CVPR 2023):
  https://github.com/facebookresearch/ConvNeXt-V2.

Third-party datasets and sample images have separate access and license terms.
The release omits raw benchmark images and checkpoints. Before publication,
verify any obligations to add original per-file license headers and any NOTICE
files required by the particular upstream source versions used.
