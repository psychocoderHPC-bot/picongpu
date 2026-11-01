import numpy as np

from neuralbunch import sampling
from neuralbunch.features import fourier_v1, feature_dim


def test_feature_shape_and_ordering():
    r = np.array([[0.1, 0.2, 0.3]])
    u = np.array([[0.0, 0.0, 1.0]])
    f = fourier_v1(r, u, 1.0, 2.0)
    assert f.shape == (1, 69) == (1, feature_dim())
    np.testing.assert_allclose(f[0, :9], [0.1, 0.2, 0.3, 0.0, 0.0, 1.0, np.sqrt(2), 1.0, 2.0])
    # first periodic feature: axis x, octave -3, sin
    assert np.isclose(f[0, 9], np.sin(0.1 * np.pi * 2 ** -3))
    assert np.isclose(f[0, 10], np.cos(0.1 * np.pi * 2 ** -3))


def test_sampling_determinism():
    a = sampling.make_sample("train", 7, "gamma10", "axis_x", "random", 0)
    b = sampling.make_sample("train", 7, "gamma10", "axis_x", "random", 0)
    c = sampling.make_sample("train", 8, "gamma10", "axis_x", "random", 0)
    for k in a.particles:
        np.testing.assert_array_equal(a.particles[k], b.particles[k])
    assert not np.array_equal(a.particles["x"], c.particles["x"])
    s1 = sampling.make_scenario_sample("val", 3, ("gamma2", "diag_p", 2, "near"), 0)
    s2 = sampling.make_scenario_sample("val", 3, ("gamma2", "diag_p", 2, "near"), 0)
    np.testing.assert_array_equal(s1.origin, s2.origin)
    assert s1.a_y == s2.a_y


def test_split_configs_are_disjoint():
    sets = {sp: set(sampling.scenario_keys(sp)) for sp in sampling.SPLITS}
    assert sets["train"] and sets["val"] and sets["test"]
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
    # deterministic mapping
    for key in list(sets["train"])[:5]:
        assert sampling.assign_split(sampling.config_key(*key)) == "train"


def test_aspects_span_full_range_continuously():
    vals = [sampling.make_sample("train", i, "gamma2", "random", "random").a_y for i in range(40)]
    assert min(vals) >= 0.1 and max(vals) <= 10.0
    assert len({round(v, 6) for v in vals}) > 35


def test_chunk_sizes_variable():
    assert sampling.chunk_sizes(10, (1, 17, 1024, 5)) == [1, 9]
    for n in (1, 17, 1024, 123):
        sizes = sampling.chunk_sizes(n, (1, 17, 1024, 5))
        assert sum(sizes) == n
        assert all(s > 0 for s in sizes)


def test_sample_chunk_preserves_total():
    s = sampling.sample_bunch("unequal_weights", seed=4, n_particles=40)
    chunks = s.chunk((1, 17, 1024, 5))
    assert sum(c.n for c in chunks) == s.n
    np.testing.assert_allclose(
        np.concatenate([c.particles["w"] for c in chunks]), s.particles["w"]
    )


def test_bunch_kinds():
    for kind in sampling.BUNCH_KINDS:
        s = sampling.sample_bunch(kind, seed=2, n_particles=32)
        assert s.n == 32
        assert np.all(np.isfinite(s.particles["x"]))
        assert np.all(s.particles["w"] >= 0)
    cs = sampling.sample_bunch("counterstreaming", seed=2, n_particles=32)
    assert np.any(cs.particles["ux"] > 0) and np.any(cs.particles["ux"] < 0)
