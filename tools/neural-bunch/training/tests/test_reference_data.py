import json
from pathlib import Path

import numpy as np

from neuralbunch import sampling
from neuralbunch.reference_data import build_dataset, load_split
from neuralbunch.reference_loader import (
    flatten_tile,
    load_standin,
    target_from_sample,
)


def test_flatten_order():
    arr = np.arange(8 * 8 * 4 * 6, dtype=np.float64).reshape(8, 8, 4, 6)
    flat = flatten_tile(arr)
    for i in (0, 3, 7):
        for j in (0, 5):
            for k in (0, 2):
                for c in range(6):
                    assert flat[c + 6 * (i + 8 * (j + 8 * k))] == arr[i, j, k, c]


def test_standin_is_marked_and_finite():
    mod = load_standin()
    assert "STAND-IN" in mod.STANDIN_WARNING
    s = sampling.sample_bunch("gaussian", seed=1, n_particles=8)
    t = target_from_sample(mod, s)
    assert t.shape == (1536,)
    assert np.all(np.isfinite(t))


def test_standin_chunk_additivity():
    mod = load_standin()
    s = sampling.sample_bunch("unequal_weights", seed=2, n_particles=40)
    full = target_from_sample(mod, s)
    total = np.zeros_like(full)
    for c in s.chunk((1, 17, 1024, 5)):
        total += target_from_sample(mod, c)
    np.testing.assert_allclose(full, total, rtol=1e-12, atol=1e-12)


def test_build_resume_and_load(tmp_path):
    m1 = build_dataset(tmp_path, "train", seed=5, n_configs=4, samples_per_config=1, shard_size=2, _allow_standin=True)
    assert m1["reference"]["kind"] == "analytic_standin"
    assert len(m1["shards"]) == 2
    hashes = [s["sha256"] for s in m1["shards"]]
    # resume must not recompute or change finished shards
    m2 = build_dataset(tmp_path, "train", seed=5, n_configs=4, samples_per_config=1, shard_size=2, _allow_standin=True)
    assert [s["sha256"] for s in m2["shards"]] == hashes
    d = load_split(tmp_path, "train")
    assert d.feats.shape == (d.src_w.shape[0], 69)
    assert d.targets.shape == (d.n_tiles, 1536)
    assert d.n_tiles == len(m2["descriptors"])
    assert np.all(np.isfinite(d.targets))
    assert np.allclose(d.src_w, -1.0)
    assert d.s0.shape == (d.n_tiles, 2)
    assert np.all(d.s0 >= 0)  # the stand-in underflows to zero at large distance
    assert all("s0" in desc for desc in m2["descriptors"])
    assert all(desc["d_min_cells"] >= 4.0 for desc in m2["descriptors"])


def test_split_datasets_are_config_disjoint(tmp_path):
    for sp, seed in (("train", 1), ("val", 2), ("test", 3)):
        build_dataset(tmp_path, sp, seed=seed, n_configs=3, samples_per_config=1, shard_size=4, _allow_standin=True)
    keys = {}
    for sp in ("train", "val", "test"):
        m = json.loads((Path(tmp_path) / sp / "manifest.json").read_text())
        keys[sp] = {json.dumps(d["config"]) for d in m["descriptors"]}
    assert not (keys["train"] & keys["val"])
    assert not (keys["train"] & keys["test"])
    assert not (keys["val"] & keys["test"])
