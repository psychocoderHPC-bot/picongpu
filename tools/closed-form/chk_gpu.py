"""Compare H100 GPU outputs vs the Python reference chain, 64 stationary pairs.

Files: e5.bin = extract o5, s80.bin = sing(8,0), s82.bin = sing(8,2).
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from cf1_core import ref_fixed, insupport_field

Z = np.zeros(3)
DELTA = np.array([
    [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
    [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])


def load(path, n):
    raw = np.fromfile(path, dtype=np.uint8)
    b = np.frombuffer(raw.tobytes()[: 3 * n * 8], dtype=np.float64).reshape(n, 3)
    c = np.frombuffer(raw.tobytes()[3 * n * 8: 3 * n * 8 + 4 * n], dtype=np.int32)
    o = np.frombuffer(raw.tobytes()[3 * n * 8 + 4 * n: 3 * n * 8 + 4 * n + 8 * n], dtype=np.float64)
    return b, c, o


def main():
    n = 64
    rel, comp, _ = load("e5.bin", n)
    print("computing refs (12,2) for 64 pairs ...", flush=True)
    refs = []
    for i in range(n):
        p = rel[i] + DELTA[comp[i]]
        refs.append(ref_fixed(p, Z, order=12, refine=2))
    print("refs done", flush=True)

    for tag, path, cpu_scheme in (("extract o5", "e5.bin", 5), ("sing (8,0)", "s80.bin", None),
                                  ("sing (8,2)", "s82.bin", None)):
        _, _, gpu = load(path, n)
        errs, errs_cpu = [], []
        for i in range(n):
            cc = int(comp[i])
            ref = refs[i]
            rc = ref[0][cc] if cc < 3 else ref[1][cc - 3]
            e = abs(gpu[i] - rc) / (1.0 + abs(rc))
            errs.append(e)
            if cpu_scheme is not None:
                p = rel[i] + DELTA[cc]
                cpu_e, cpu_b, _ = insupport_field(p, Z, "P2", cpu_scheme)
                cpu_c = cpu_e[cc] if cc < 3 else cpu_b[cc - 3]
                errs_cpu.append(abs(cpu_c - rc) / (1.0 + abs(rc)))
        line = f"{tag}: GPU vs ref(12,2) worst {max(errs):.3e}"
        if cpu_scheme is not None:
            line += f"   CPU P2 o{cpu_scheme} vs ref worst {max(errs_cpu):.3e}"
        print(line, flush=True)
        # argmax detail
        i = int(np.argmax(errs))
        cc = int(comp[i])
        ref = refs[i]
        rc = ref[0][cc] if cc < 3 else ref[1][cc - 3]
        print(f"   worst pair {i}: c={cc} p={rel[i] + DELTA[cc]} gpu={gpu[i]:.8g} ref={rc:.8g} "
              f"singular={np.max(np.abs(rel[i] + DELTA[cc])) < 1.5}")


if __name__ == "__main__":
    main()
