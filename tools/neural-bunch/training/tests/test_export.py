import json
import shutil

import numpy as np
import pytest

from neuralbunch.export import ArtifactError, load_artifact, write_artifact
from neuralbunch.inspect import inspect_artifact
from neuralbunch.model import ModelConfig, NeuralBunchModel

torch = pytest.importorskip("torch")


def make_model(seed=0):
    torch.manual_seed(seed)
    return NeuralBunchModel(ModelConfig(hidden=(12, 10), latent=7))


def test_export_roundtrip(tmp_path):
    model = make_model()
    prefix = tmp_path / "model"
    meta = write_artifact(model, prefix, code_sha="a" * 40)
    assert (tmp_path / "model.json").is_file() and (tmp_path / "model.bin").is_file()
    raw = json.loads((tmp_path / "model.json").read_text())
    assert raw["format"] == "picongpu-neural-bunch"
    assert raw["version"] == 1
    assert raw["features"]["dim"] == 69
    assert raw["tile"]["components"] == ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
    # 64-byte alignment, monotonic, non-overlapping
    last = -1
    for spec in raw["tensors"]:
        assert spec["offset_bytes"] % 64 == 0
        assert spec["offset_bytes"] >= last
        last = spec["offset_bytes"] + int(np.prod(spec["shape"])) * 4
    assert raw["artifact"]["bytes"] == (tmp_path / "model.bin").stat().st_size
    meta2, tensors = load_artifact(prefix)
    assert meta2["artifact"]["sha256"] == meta["artifact"]["sha256"]
    for name, arr in model.artifact_tensors():
        np.testing.assert_allclose(tensors[name], arr.astype(np.float32), rtol=1e-7, atol=1e-8)
    rep = inspect_artifact(prefix)
    assert rep["valid"] is True


def test_decoder_bank_order_and_shapes(tmp_path):
    model = make_model()
    meta = write_artifact(model, tmp_path / "m", code_sha="b" * 40)
    nz, ny = len(meta["decoder"]["axes"]["a_z"]), len(meta["decoder"]["axes"]["a_y"])
    assert len(meta["decoder"]["bank"]) == nz * ny
    assert meta["decoder"]["bank"][0] == "dec.0.0.w"
    assert meta["decoder"]["bank"][1] == "dec.0.1.w"  # a_y varies fastest
    _, tensors = load_artifact(tmp_path / "m")
    for iz in range(nz):
        for iy in range(ny):
            assert tensors[f"dec.{iz}.{iy}.w"].shape == (7, 1536)


def test_inspect_rejects_corrupted_payload(tmp_path):
    model = make_model()
    write_artifact(model, tmp_path / "good", code_sha="c" * 40)
    shutil.copy(tmp_path / "good.json", tmp_path / "bad.json")
    bin_data = bytearray((tmp_path / "good.bin").read_bytes())
    bin_data[100] ^= 0xFF
    (tmp_path / "bad.bin").write_bytes(bytes(bin_data))
    meta = json.loads((tmp_path / "bad.json").read_text())
    meta["artifact"]["file"] = "bad.bin"
    (tmp_path / "bad.json").write_text(json.dumps(meta))
    with pytest.raises(ArtifactError):
        inspect_artifact(tmp_path / "bad")


def test_inspect_rejects_schema_and_size(tmp_path):
    model = make_model()
    write_artifact(model, tmp_path / "m", code_sha="d" * 40)
    (tmp_path / "m.bin").write_bytes((tmp_path / "m.bin").read_bytes()[:-8])
    with pytest.raises(ArtifactError):
        load_artifact(tmp_path / "m")

    meta = json.loads((tmp_path / "m.json").read_text())
    meta["tensors"][0]["offset_bytes"] = 3
    (tmp_path / "m.json").write_text(json.dumps(meta))
    with pytest.raises(ArtifactError):
        load_artifact(tmp_path / "m")


def test_export_refuses_wrong_geometry(tmp_path):
    class Bad:
        cfg = ModelConfig(hidden=(8,), latent=5)

        def artifact_tensors(self):
            m = make_model()
            return m.artifact_tensors()

    with pytest.raises(ValueError):
        write_artifact(Bad(), tmp_path / "bad", code_sha="e" * 40)


def test_torch_numpy_parity(tmp_path):
    from neuralbunch.parity import compare_artifact

    write_artifact(make_model(3), tmp_path / "p", code_sha="f" * 40)
    res = compare_artifact(tmp_path / "p", n_particles=129, n_trials=2)
    assert res["passed"], res
    assert res["max_relative_error"] < 1e-3
