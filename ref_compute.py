"""Launcher: one isolated subprocess per reference call, 6 parallel, retries.
Robust to the intermittent semop/BLAS deadlock (each subprocess is fresh)."""
import os, sys, time, subprocess
from concurrent.futures import ThreadPoolExecutor

WORK = os.path.dirname(os.path.abspath(__file__))
GAMMAS = [1, 10, 100]
ASPECTS = ["0.25", "4.0"]
DIRV = [0.6, 0.5, 0.62]

# need POS count; import lib (numpy single-thread not critical here)
sys.path.insert(0, WORK)
import t123lib as L
NPOS = len(L.positions(n_dir=12, radii=(0.5, 1.0), face=1.49))

ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")

tasks = [(g, a, i) for g in GAMMAS for a in ASPECTS for i in range(NPOS)]
pending = [t for t in tasks if not os.path.exists(f"{WORK}/ref1_g{t[0]}_a{t[1]}_i{t[2]}.npz")]
print(f"total={len(tasks)} pending={len(pending)}", flush=True)

def work(t):
    g, a, i = t
    fn = f"{WORK}/ref1_g{g}_a{a}_i{i}.npz"
    for attempt in range(4):
        try:
            r = subprocess.run([sys.executable, "ref_one.py", str(g), a, str(i)],
                               cwd=WORK, env=ENV, capture_output=True, timeout=150)
            if r.returncode == 0 and os.path.exists(fn):
                return t, 0
        except Exception:
            pass
        time.sleep(2)
    return t, 1

t0 = time.time()
done = 0
with ThreadPoolExecutor(max_workers=6) as ex:
    for (t, ok) in ex.map(work, pending):
        done += 1
        if not ok:
            print(f"FAILED after retries: g={t[0]} a={t[1]} i={t[2]}", flush=True)
        elif done % 8 == 0:
            print(f"  {done}/{len(pending)} refs  ({time.time()-t0:.0f}s)", flush=True)
print(f"ALL DONE in {time.time()-t0:.0f}s, {done} computed", flush=True)
