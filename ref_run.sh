#!/bin/bash
# Launcher: one fresh single-threaded subprocess per reference call, 4 parallel, retries.
# Pure bash (no numpy in the driver) so the semop/BLAS deadlock can't hit the launcher.
cd /home/ubuntu/pr24/pr24w123 || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
rm -f failures.txt
run_one() {
  local g="$1" a="$2" i="$3"
  local f="ref1_g${g}_a${a}_i${i}.npz"
  [ -f "$f" ] && return 0
  local attempt
  for attempt in 1 2 3 4 5; do
    python3 ref_one.py "$g" "$a" "$i" >/dev/null 2>&1 && [ -f "$f" ] && return 0
    sleep 2
  done
  echo "FAILED g=$g a=$a i=$i" >> failures.txt
  return 1
}
export -f run_one
: > tasks.txt
for g in 1 10 100; do
  for a in 0.25 4.0; do
    for i in $(seq 0 31); do echo "$g $a $i"; done
  done
done > tasks.txt
xargs -P 4 -n 3 bash -c 'run_one "$0" "$1" "$2"' < tasks.txt
echo "ALL REFS DONE: $(ls ref1_*.npz 2>/dev/null | wc -l)/192  failures=$(wc -l < failures.txt 2>/dev/null)"
