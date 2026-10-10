#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HLA_CKPT:?Set HLA_CKPT to your trained HLA Lightning checkpoint}"
python test.py configs/data/megadepth_test_1500.py configs/hla/outdoor/test.py \
  --ckpt_path "$HLA_CKPT" --batch_size 1 --num_workers 2 \
  --dump_dir outputs/megadepth_eval
