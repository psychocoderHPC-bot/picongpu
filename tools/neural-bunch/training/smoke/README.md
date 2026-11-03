# Smoke fixture (plumbing only, NOT a trained model)

This directory is a checked-in plumbing fixture: a tiny random-weight model
written through the real export path, used to exercise loading, inspection,
validation reporting and parity. It is **not a trained model** and its numbers
are **not** physics validation. It was produced before the reference was fixed;
at that time the reference `tile_field` raised a reshape error
`cannot reshape array of size 9216 into shape (8,8,4,6)` and a single-source
call later exceeded the timeout). The dataset is therefore generated with
`--allow-standin` and is labeled `reference.kind = "analytic_standin"`.

**These errors are not physics validation.** They only prove that sampling,
sharding, training, export, inspection, validation reporting and parity run
end to end.

Artifact: `model.json` + `model.bin` (2,472,896 bytes, SHA-256
`484b35b09d182adf3b20d36cf2e7698e99a2586bb1e6151b36ffcd82198dded1`,
code SHA `838b44090f2d2480a3fbac46dfef5b163a69fd60`).
Inspect accepts it. The test suite creates a corrupted copy (one flipped byte)
and checks that it is rejected with `payload sha256 mismatch`. Torch/numpy parity max relative error
is `4.67e-7`.

Training configuration: 24 train / 24 val / 24 test cases, hidden 32,32, latent
16, 600 steps, seed 0, AdamW lr 1e-3.

Validation errors against the stand-in:

| split | group | mean rel RMS | p99 | max | absolute RMS |
|-------|-------|--------------|-----|-----|--------------|
| train | E | 0.911 | 0.9996 | 1.000 | 0.165 |
| train | B | 37.79 | 70.31 | 70.81 | 0.0059 |
| val   | E | 1.800 | 6.921 | 7.365 | 1.270 |
| val   | B | 2893 | 13439 | 14390 | 1.172 |
| test  | E | 2.606 | 20.49 | 24.20 | 0.426 |
| test  | B | 1523 | 9143 | 10002 | 0.368 |

These errors are large because the model, dataset and step budget are tiny.
They are not evidence of final model quality.

Reproduce:

```bash
./venv/bin/python -m neuralbunch reference --out smoke/data --split all \
  --n-configs 12 --samples-per-config 2 --shard-size 4 --allow-standin --seed 0
./venv/bin/python -m neuralbunch train --data smoke/data --out smoke/run \
  --hidden 32,32 --latent 16 --max-steps 600 --eval-every 100 \
  --batch-tiles 8 --lr 1e-3 --seed 0 --export smoke/model \
  --validation-report smoke/validation_report.json
./venv/bin/python -m neuralbunch inspect smoke/model
./venv/bin/python -m neuralbunch parity --artifact smoke/model --particles 257 --trials 3
```
