import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralbunch.train import _metric_block, loss_and_terms  # noqa: E402


def test_preregistered_normalization_and_absolute_cases():
    # case 0: zero target -> denominator must be S0 (absolute_normalized)
    # case 1: strong target -> denominator must be ||A_ref||
    target = torch.zeros(2, 1536, dtype=torch.float64)
    target[1, :768] = 0.1  # E components only, flattened order is irrelevant here
    pred = target.clone()
    pred[0, :3] = 0.01
    pred[0, 3:6] = 0.02
    pred[1, :768] += 0.001
    s0 = torch.tensor([[0.5, 0.5], [0.001, 0.001]], dtype=torch.float64)
    rel, denom, absn, refn, summary = _metric_block(
        pred.reshape(2, 256, 6), target.reshape(2, 256, 6), s0[:, 0], "E"
    )
    assert absn[0] and not absn[1]
    assert denom[0] == pytest.approx(0.5)
    assert denom[1] == pytest.approx(np.sqrt(768) * 0.1)
    assert summary["n_absolute_normalized"] == 1
    assert summary["n"] == 2
    # no silent exclusion
    assert np.all(np.isfinite(rel))


def test_loss_uses_the_same_preregistered_scale():
    target = torch.zeros(1, 1536)
    pred = torch.zeros(1, 1536)
    pred.reshape(1, 256, 6)[..., :3] = 0.25  # E only
    s0 = torch.tensor([[0.5, 0.5]])
    loss, le, lb = loss_and_terms(pred, target, s0)
    expected_e = float(768 * 0.25 ** 2 / 0.25)
    assert le == pytest.approx(expected_e, rel=1e-5)
    assert lb == pytest.approx(0.0, abs=1e-12)
    assert loss == pytest.approx(expected_e, rel=1e-5)
