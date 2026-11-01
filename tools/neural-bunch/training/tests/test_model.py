import numpy as np
import pytest

from neuralbunch import sampling
from neuralbunch.features import fourier_v1
from neuralbunch.model import ModelConfig, NeuralBunchModel

torch = pytest.importorskip("torch")


def small_cfg():
    return ModelConfig(hidden=(16, 16), latent=8)


def features_for(sample):
    p = sample.particles
    pos = np.stack([p["x"], p["y"], p["z"]], axis=1)
    u = np.stack([p["ux"], p["uy"], p["uz"]], axis=1)
    r = (pos - sample.origin) / sample.dx
    return fourier_v1(r, u, sample.a_y, sample.a_z), -p["w"]


def seeded_model(cfg=None):
    torch.manual_seed(0)
    m = NeuralBunchModel(cfg or small_cfg()).double()
    m.eval()
    return m


def test_chunk_additivity_exact():
    s = sampling.sample_bunch("unequal_weights", seed=11, n_particles=40)
    f, w = features_for(s)
    m = seeded_model()
    full = m.forward_tile(torch.tensor(f), torch.tensor(w), s.a_y, s.a_z)
    chunks = s.chunk((1, 17, 1024, 5))
    total = 0.0
    for c in chunks:
        fc, wc = features_for(c)
        total = total + m.forward_tile(torch.tensor(fc), torch.tensor(wc), c.a_y, c.a_z)
    np.testing.assert_allclose(full.detach().numpy(), total.detach().numpy(), rtol=1e-12, atol=1e-12)


def test_permutation_invariance():
    s = sampling.sample_bunch("gaussian", seed=3, n_particles=33)
    f, w = features_for(s)
    perm = np.random.default_rng(5).permutation(len(w))
    m = seeded_model()
    a = m.forward_tile(torch.tensor(f), torch.tensor(w), s.a_y, s.a_z)
    b = m.forward_tile(torch.tensor(f[perm]), torch.tensor(w[perm]), s.a_y, s.a_z)
    np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy(), rtol=1e-11, atol=1e-11)


def test_zero_weight_padding_is_masked():
    s = sampling.sample_bunch("gaussian", seed=6, n_particles=9)
    f, w = features_for(s)
    m = seeded_model()
    base = m.forward_tile(torch.tensor(f), torch.tensor(w), s.a_y, s.a_z)
    # pad with NaN/Inf rows carrying zero weight; they must never enter the encoder
    fpad = np.concatenate([f, np.full((5, f.shape[1]), np.nan)])
    wpad = np.concatenate([w, np.zeros(5)])
    out = m.forward_tile(torch.tensor(fpad), torch.tensor(wpad), s.a_y, s.a_z)
    np.testing.assert_allclose(base.detach().numpy(), out.detach().numpy(), rtol=1e-13, atol=1e-13)


def test_zero_charge_gives_zero_field():
    s = sampling.sample_bunch("flat_top", seed=9, n_particles=5)
    f, _ = features_for(s)
    m = seeded_model()
    out = m.forward_tile(torch.tensor(f), torch.zeros(len(f)), s.a_y, s.a_z)
    assert np.all(out.detach().numpy() == 0.0)


def test_bilinear_anchor_reproduces_bank_entry():
    from neuralbunch.model import interp_indices_weights

    cfg = small_cfg()
    m = seeded_model(cfg)
    bank = m.decoder.bank.detach().numpy()
    for iz, az in enumerate(cfg.a_z_anchors):
        for iy, ay in enumerate(cfg.a_y_anchors):
            idx, w = interp_indices_weights(ay, az, cfg.a_y_anchors, cfg.a_z_anchors)
            target = iz * len(cfg.a_y_anchors) + iy
            interp = sum(w[k] * bank[int(idx[k])] for k in range(4))
            np.testing.assert_allclose(interp, bank[target], atol=1e-14)


def test_config_rejects_bad_geo():
    with pytest.raises(ValueError):
        ModelConfig(octaves=(-2, 6)).validate()
    with pytest.raises(ValueError):
        ModelConfig(a_y_anchors=[10.0, 0.1]).validate()
    with pytest.raises(ValueError):
        ModelConfig(a_y_anchors=[0.01, 1.0]).validate()
