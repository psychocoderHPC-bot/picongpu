#!/usr/bin/env bash
# Generate the six T150 near-field tables (gamma 1/10/100 x ay 0.25/4, az 1) with the
# vendored GPU generator (build_table.py). One table per (gamma, aspect); the velocity is
# exact per table (direction 0.6/0.5/0.62, magnitude sqrt(gamma^2 - 1); irrelevant for
# gamma 1, which is stationary).
#
# Requirements: a Python venv with numpy + cuPy (the generator runs the tensor-product
# reduction on one GPU), and one GPU per build. On rosi_v100 the builds must run
# strictly serially with `env -u LD_PRELOAD` (the fakeroot SysV semaphore preloading
# crashes cuPy builds with "semop(1): Invalid argument").
#
# Usage: OUT=<table dir> GPU=<index> PY=<venv python> ./generate.sh
set -euo pipefail
OUT="${OUT:-table}"
GPU="${GPU:-1}"
PY="${PY:-python3}"
mkdir -p "$OUT"
DIR="0.6 0.5 0.62"
GRID=(--rmax 13 --h-near 0.2 --h-far 0.5 --r-near 4 --order-near 12 --refine-near 3 --order-far 8 --batch 512)
for gamma in 1 10 100; do
  for ay in 0.25 4; do
    tag="gamma${gamma}_ay${ay}_az1"
    if [ -s "${OUT}/nk_${tag}.bin" ]; then
      echo "SKIP ${tag} (exists)"
      continue
    fi
    CUDA_VISIBLE_DEVICES="$GPU" env -u LD_PRELOAD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$PY" "$(dirname "$0")/build_table.py" --gamma "$gamma" --dir $DIR \
      --cells 1 "$ay" 1 "${GRID[@]}" --out "${OUT}/nk"
  done
done
ls -la "$OUT"/nk_gamma*_ay*_az1.bin
