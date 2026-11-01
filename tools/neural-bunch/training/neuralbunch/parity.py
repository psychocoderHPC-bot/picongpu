"""Torch vs pure-numpy forward parity for an exported artifact."""

from __future__ import annotations

import json

import numpy as np

from .export import load_artifact
from .model import ModelConfig, NeuralBunchModel, NumpyArtifactModel


def config_from_metadata(meta) -> ModelConfig:
    layers = meta["encoder"]["layers"]
    hidden = [int(layer["out"]) for layer in layers[:-1]]
    latent = int(layers[-1]["out"])
    kind = meta["features"].get("kind", "fourier_v1")
    oct = tuple(meta["features"].get("octaves", (-3, 6)))
    return ModelConfig(
        hidden=hidden,
        latent=latent,
        feature_kind=kind,
        octaves=oct if kind == "fourier_v1" else (-3, 6),
        probe_octaves=oct if kind == "fourier_probe_v1" else (-3, 1),
        a_y_anchors=meta["decoder"]["axes"]["a_y"],
        a_z_anchors=meta["decoder"]["axes"]["a_z"],
    )


def torch_model_from_artifact(meta, tensors):
    import torch

    cfg = config_from_metadata(meta)
    model = NeuralBunchModel(cfg)
    sd = {}
    for i, layer in enumerate(meta["encoder"]["layers"]):
        sd[f"encoder.linears.{i}.weight"] = torch.tensor(np.asarray(tensors[layer["weight"]]).T, dtype=torch.float32)
        sd[f"encoder.linears.{i}.bias"] = torch.tensor(np.asarray(tensors[layer["bias"]]), dtype=torch.float32)
    bank = np.stack([np.asarray(tensors[name]) for name in meta["decoder"]["bank"]], axis=0)
    sd["decoder.bank"] = torch.tensor(bank, dtype=torch.float32)
    sd["encoder.feature_gate"] = torch.ones(int(meta["features"]["dim"]), dtype=torch.float32)
    model.load_state_dict(sd)
    # exported encoder weights already contain the fixed input conditioning and
    # the learned feature gate
    model.encoder.input_scale.fill_(1.0)
    model.encoder.feature_gate.data.fill_(1.0)
    model.eval()
    return model


def compare_artifact(prefix, n_particles=257, n_trials=3, seed=1234):
    meta, tensors = load_artifact(prefix)
    np_model = NumpyArtifactModel(meta, tensors)
    t_model = torch_model_from_artifact(meta, tensors)
    rng = np.random.default_rng(seed)
    worst = 0.0
    per_trial = []
    for t in range(n_trials):
        n = int(n_particles)
        feats = rng.normal(size=(n, meta["features"]["dim"]))
        # mimic realistic feature scales for the periodic components
        src_w = rng.uniform(-1, 1, size=n)
        a_y = float(10 ** rng.uniform(-1, 1))
        a_z = float(10 ** rng.uniform(-1, 1))
        np_out = np_model.forward_tile(feats, src_w, a_y, a_z)
        with __import__("torch").no_grad():
            t_out = (
                t_model.forward_tile(
                    __import__("torch").tensor(feats, dtype=__import__("torch").float32),
                    __import__("torch").tensor(src_w, dtype=__import__("torch").float32),
                    a_y,
                    a_z,
                )
                .double()
                .numpy()
            )
        rel = float(np.linalg.norm(t_out - np_out) / max(np.linalg.norm(np_out), 1e-12))
        per_trial.append(rel)
        worst = max(worst, rel)
    return {"max_relative_error": worst, "per_trial": per_trial, "passed": worst < 1e-3}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Torch/numpy parity for an artifact")
    ap.add_argument("artifact")
    ap.add_argument("--particles", type=int, default=257)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args(argv)
    res = compare_artifact(args.artifact, args.particles, args.trials)
    print(json.dumps(res, indent=2))
    return 0 if res["passed"] else 1
