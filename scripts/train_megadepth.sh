#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${HLA_BACKBONE_WEIGHTS:-}" ]]; then
  echo "Set HLA_BACKBONE_WEIGHTS to the pretrained ConvNeXt V2 nano backbone checkpoint." >&2
  exit 2
fi
python train.py configs/data/megadepth_trainval_832.py configs/hla/outdoor/final.py \
  --exp_name hla_megadepth --batch_size 2 --num_workers 4 \
  --gpus 1 --max_epochs 30
