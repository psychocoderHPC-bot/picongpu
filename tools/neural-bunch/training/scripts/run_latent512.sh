#!/bin/bash
set -euo pipefail
DATA="${DATA:-data/dataset}"
exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ${PYTHON:-python} -m neuralbunch train \
  --data "$DATA" --out runs/latent512 \
  --hidden 512,512 --latent 512 --optimizer adamw --lr 3e-4 --weight-decay 1e-4 \
  --batch-tiles 64 --max-steps 20000 --eval-every 2000 --seed 0 \
  --lr-schedule cosine --force-weight 0.1 --time-budget-s 2400 \
  --device cuda:1 --eval-splits train,val --export runs/latent512/artifact
