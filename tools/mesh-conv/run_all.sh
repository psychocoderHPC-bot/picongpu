#!/bin/bash
# Build and run on rosi_v100 (CUDA 12.6, V100 sm_70).  GPU 3 only.
# The FFT comes from alpakaVendor at the PIConGPU pin; only the portable
# alpaka::fft API is used.  alpakaVendor's CUDA backend maps to cuFFT, so the
# toolkit's cuFFT library must be on the link line.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
HERE=$(cd "$(dirname "$0")" && pwd)
ALPAKAV_DIR=${ALPAKAV_DIR:-$HOME/t152/alpakaVendor}
ALPAKAV_PIN=${ALPAKAV_PIN:-af24cc7b2bd125056e9889e270d3af7bdf972c6f}
REF_DIR=${REF_DIR:-$HOME/pr24/pr24w53-ref}

if [ ! -f "$ALPAKAV_DIR/include/alpaka/fft.hpp" ]; then
  git clone https://github.com/psychocoderHPC-bot/alpakaVendor.git "$ALPAKAV_DIR"
  git -C "$ALPAKAV_DIR" checkout "$ALPAKAV_PIN"
  git -C "$ALPAKAV_DIR" submodule update --init --recursive
fi
test "$(git -C "$ALPAKAV_DIR" rev-parse HEAD)" = "$ALPAKAV_PIN"

if [ ! -f "$HERE/indfield.py" ]; then
  cp "$REF_DIR/indfield.py" "$HERE/indfield.py"
fi

cd "$HERE"
"$NVCC" -O3 -arch=sm_70 -std=c++20 \
  -I"$ALPAKAV_DIR/include" -I"$ALPAKAV_DIR/extern/alpaka3/include" \
  meshconv.cu -lcufft -o meshconv
echo "built meshconv (alpakaVendor $ALPAKAV_PIN)"

./meshconv --mode info
./meshconv --mode validate > validate_double.csv
python3 compare_ref.py validate_double.csv --out validation_double.json | tee validation_double.txt

./meshconv --mode validate --fft float > validate_float.csv
python3 compare_ref.py validate_float.csv --out validation_float.json | tee validation_float.txt

./meshconv --mode nopad | tee nopad.txt

./meshconv --mode bench > bench_double.csv
./meshconv --mode bench --fft float > bench_float.csv
echo DONE
