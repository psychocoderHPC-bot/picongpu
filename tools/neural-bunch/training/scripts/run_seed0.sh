#!/bin/bash
set -euo pipefail
DATA="${DATA:-data/dataset}"
while [ "$(ls "$DATA"/val/shard_*.npz 2>/dev/null | wc -l)" -lt 43 ]; do sleep 5; done
sleep 2
exec env PYTHONUNBUFFERED=1 ${PYTHON:-python} -m neuralbunch train \
  --data "$DATA" --out runs/seed0 \
  --hidden 512,512 --latent 256 --optimizer adamw --lr 3e-4 --weight-decay 1e-4 \
  --batch-tiles 64 --max-steps 100000 --eval-every 2000 --seed 0 \
  --force-weight 0.1 --time-budget-s 5400 --device cuda:1 --eval-splits train,val \
  --export runs/seed0/artifact
