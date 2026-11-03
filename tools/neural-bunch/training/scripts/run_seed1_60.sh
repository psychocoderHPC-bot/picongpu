#!/bin/bash
set -euo pipefail
DATA="${DATA:-data/dataset}"
exec env PYTHONUNBUFFERED=1 ${PYTHON:-python} -m neuralbunch train \
  --data "$DATA" --out runs/seed1_cos60 \
  --hidden 512,512 --latent 256 --optimizer adamw --lr 3e-4 --weight-decay 1e-4 \
  --batch-tiles 64 --max-steps 60000 --eval-every 2000 --seed 1 \
  --lr-schedule cosine --force-weight 0.1 --time-budget-s 3600 \
  --device cuda:0 --eval-splits train,val --export runs/seed1_cos60/artifact
