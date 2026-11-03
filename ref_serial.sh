#!/bin/bash
# Serial reference builder: one fresh single-threaded subprocess at a time, retries.
cd /home/ubuntu/pr24/pr24w123 || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
done=0; failed=0
t0=$(date +%s)
for g in 1 10 100; do
  for a in 0.25 4.0; do
    for i in $(seq 0 31); do
      f="ref1_g${g}_a${a}_i${i}.npz"
      if [ -f "$f" ]; then done=$((done+1)); continue; fi
      ok=0
      for attempt in 1 2 3 4 5 6; do
        python3 ref_one.py "$g" "$a" "$i" >/dev/null 2>&1
        [ -f "$f" ] && { ok=1; break; }
        sleep 2
      done
      done=$((done+1))
      if [ $ok -eq 0 ]; then failed=$((failed+1)); echo "FAILED g=$g a=$a i=$i" >> failures.txt; fi
      if [ $((done % 8)) -eq 0 ]; then echo "  $done/192 refs  elapsed=$(( ($(date +%s)-t0) ))s"; fi
    done
  done
done
echo "SERIAL REFS DONE: $(ls ref1_*.npz 2>/dev/null | wc -l)/192  failed=$failed  elapsed=$(( ($(date +%s)-t0) ))s"
