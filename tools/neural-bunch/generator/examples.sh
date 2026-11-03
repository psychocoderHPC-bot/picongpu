#!/usr/bin/env bash
# Generate the required physical bunch cases in both backends, inspect them and
# compare HDF5 vs ADIOS2 arrays.  Generated data stays outside the checkout.
#
#   OUT=./neural-bunch-data ./examples.sh
#
# Override the interpreter with PYTHON=... (needs openpmd_api 0.17.1).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
OUT="${OUT:-./neural-bunch-data}"
WORK="${WORK:-$(dirname "$OUT")}"
mkdir -p "$OUT"

if ! "$PY" -c 'import openpmd_api, numpy, h5py' >/dev/null 2>&1; then
  echo "error: $PY lacks openpmd_api/numpy/h5py (set PYTHON=...)" >&2
  exit 2
fi

GEN=("$PY" "$HERE/generate_bunch.py")
INS=("$PY" "$HERE/inspect_bunch.py")

# name|seed|generator arguments
CASES=(
  "stationary|1001|--model stationary --n 4096"
  "beta001|1002|--model drift --beta 0.01 --direction 1,0,0 --n 4096"
  "beta01|1003|--model drift --beta 0.1 --direction 1,0,0 --n 4096"
  "gamma1|1013|--model gamma --gamma 1 --direction 1,0,0 --n 4096"
  "gamma2|1004|--model gamma --gamma 2 --direction 1,0,0 --n 4096"
  "gamma10|1005|--model gamma --gamma 10 --direction 1,0,0 --n 4096"
  "gamma100|1006|--model gamma --gamma 100 --direction 1,0,0 --n 4096"
  "offaxis|1007|--model drift --beta 0.05 --direction 0.6,0.8,0 --n 4096 --centroid 20 44 32 --sigma 4 5 4"
  "mixed|1008|--model mixed --n 4096 --mix-spec [{\"fraction\":0.5,\"beta\":[0.01,0,0]},{\"fraction\":0.3,\"beta\":[0,0.1,0]},{\"fraction\":0.2,\"beta\":[-0.05,0,0.02]}]"
  "counterstreaming|1009|--model counterstreaming --beta 0.3 --axis x --n 4096"
  "unequal|1010|--model stationary --n 4096 --weights two-tier --weight-ratio 2.5 --weight-fraction 0.4"
  "tiny257|1011|--model stationary --n 257 --grid 32 32 32 --centroid 16 16 16 --sigma 3 3 3"
  "chunks1025|1012|--model drift --beta 0.02 --n 1025 --chunks 3"
)

for entry in "${CASES[@]}"; do
  IFS='|' read -r name seed extra <<<"$entry"
  echo "=== $name (seed $seed) ==="
  # shellcheck disable=SC2086
  for backend in hdf5 adios2; do
    "${GEN[@]}" --output "$OUT/$name" --backend "$backend" --seed "$seed" $extra
  done
  "${INS[@]}" "$OUT/$name.h5" --compare "$OUT/$name.bp5"
  echo
done

echo "=== generated files ==="
"$PY" - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
rows = []
for name in sorted(os.listdir(out)):
    if name.endswith(".manifest.json"):
        path = os.path.join(out, name)
        manifest = json.load(open(path))
        rows.append((manifest["parameters"]["model"], os.path.basename(manifest["series"]),
                     manifest["backend"], manifest["particles"], manifest["file_bytes_total"]))
for model, series, backend, particles, size in sorted(rows):
    print(f"{series:28s} {backend:7s} n={particles:6d} {size:10d} bytes  model={model}")
PY
