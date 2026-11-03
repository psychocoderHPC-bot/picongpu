#!/bin/bash
# Build and run on rosi_v100 (CUDA 12.6, V100 sm_70).  GPU 2 only.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
export PATH=/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")"

nvcc -O3 -arch=sm_70 -std=c++17 meshconv.cu -lcufft -o meshconv
echo "built meshconv"

./meshconv --mode validate > validate_double.csv
python3 compare_ref.py validate_double.csv --out validation_double.json | tee validation_double.txt

./meshconv --mode validate --fft float > validate_float.csv
python3 compare_ref.py validate_float.csv --out validation_float.json | tee validation_float.txt

./meshconv --mode nopad | tee nopad.txt

./meshconv --mode bench > bench_double.csv
./meshconv --mode bench --fft float > bench_float.csv
echo DONE
