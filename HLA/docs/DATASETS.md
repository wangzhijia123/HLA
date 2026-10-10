# Dataset organization

Raw MegaDepth and ScanNet images/depth maps are **not included**. They are
provided by the corresponding dataset owners and may require license agreement.
Only limited publicly shared evaluation-index metadata is in `assets/`.

```text
data/
  megadepth/
    train/          MegaDepth/D2-Net preprocessed image and depth tree
    test/           MegaDepth evaluation images
    index/          scene_info_0.1_0.7, scene_info_val_1500, trainvaltest_list
  scannet/
    test/           ScanNet images/depth and scene subfolders
```

Data-specific config paths are defined in `configs/data/`. Adjust the symlink
structure or edit those config files to match your local storage layout.
Do not commit raw benchmark images, restricted scans or access tokens to GitHub.

Dataset owners / access information:

- MegaDepth: https://www.cs.cornell.edu/projects/megadepth/
- HPatches: https://github.com/hpatches/hpatches-dataset
- ScanNet: https://www.scan-net.org/ScanNet/
- LoFTR data splits and indices: https://github.com/zju3dv/LoFTR

Training scene-index archives must be downloaded separately and installed under
`data/megadepth/index/`. The `assets/` evaluation metadata may cover only a
limited scene subset and should not be described as a complete standalone dataset.
