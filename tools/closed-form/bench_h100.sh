#!/bin/bash
# CF1 step 5b: H100 benchmark matrix for the near-field kernels.
set -u
cd ~/cf1-20261102
N1M=1048576   # 2^20
N256K=262144  # 2^18
run() { echo "### $*"; ./gpu_extract "$@" --iters 4 2>&1 | grep -E "bench:|DONE|self-check|xcheck"; }

echo "===== stationary gamma=1 ====="
run --kernel extract --band in   --order 4 --npairs $N1M
run --kernel extract --band in   --order 5 --npairs $N1M
run --kernel extract --band in   --order 6 --npairs $N1M
run --kernel sing    --band in   --order 8 --refine 0 --npairs $N1M
run --kernel sing    --band in   --order 8 --refine 1 --npairs $N256K
run --kernel extract --band band --order 5 --npairs $N1M
run --kernel extract --band band --order 6 --npairs $N1M
run --kernel sing    --band band --order 8 --refine 0 --npairs $N1M
echo "===== moving gamma=10 axial (beta || z) ====="
run --kernel extract --band in   --order 5 --gamma 10 --npairs $N1M
run --kernel extract --band in   --order 6 --gamma 10 --npairs $N1M
run --kernel extract --band in   --order 8 --gamma 10 --npairs $N256K
run --kernel sing    --band in   --order 8 --refine 0 --gamma 10 --npairs $N1M
run --kernel sing    --band in   --order 8 --refine 1 --gamma 10 --npairs $N256K
echo "ALL_DONE"
