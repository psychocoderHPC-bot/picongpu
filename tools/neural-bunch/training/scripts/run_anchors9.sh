#!/bin/bash
set -euo pipefail
DATA="${DATA:-data/dataset}"
exec env PYTHONUNBUFFERED=1 ${PYTHON:-python} -m neuralbunch train \
  --data "$DATA" --out runs/anchor9_256 \
  --hidden 512,512 --latent 256 --optimizer adamw --lr 3e-4 --weight-decay 1e-4 \
  --batch-tiles 64 --max-steps 30000 --eval-every 2000 --seed 0 \
  --decoder-anchors 9 --lr-schedule cosine --force-weight 0.1 --time-budget-s 2400 \
  --device cuda:1 --eval-splits train,val --export runs/anchor9_256/artifact
