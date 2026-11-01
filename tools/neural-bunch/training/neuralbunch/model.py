"""Torch modules and a pure-numpy forward for artifact format v1.

Architecture (fixed by the orchestrator):

* additive per-source encoder ``h_theta``: a plain SiLU MLP on ``fourier_v1``;
* decoder ``D(a_y, a_z)``: a bank of linear matrices on a log-spaced aspect
  grid, bilinearly interpolated in ``log10(a)`` (clamped at the outer anchors);
* ``z = sum_i (q_i/e) h_theta(...)`` and ``F_tilde = D(a_y, a_z) z``, no bias.

``q_i/e = -w_i`` per MODEL-FORMAT.md, so callers pass the positive weight
``w_i`` and the encoder contribution is multiplied by ``-w_i``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from .features import (DEFAULT_OCTAVES, DENSE_KIND, PROBE_KIND, PROBE_OCTAVES,
                       feature_dim, feature_scale)

TILE = (8, 8, 4)
COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
OUT_DIM = TILE[0] * TILE[1] * TILE[2] * len(COMPONENTS)  # 1536
DEFAULT_AXES = tuple(np.logspace(np.log10(0.1), np.log10(10.0), 5).tolist())

#: Fixed per-feature scale used only for conditioning the MLP.  It is baked
#: into the exported first-layer weight, so the artifact still consumes raw
#: feature vectors.  ``feature_scale(kind)`` selects the scale for the kind.
_FEATURE_SCALE = feature_scale("fourier_v1")


def default_axes():
    return [list(DEFAULT_AXES), list(DEFAULT_AXES)]


@dataclass
class ModelConfig:
    hidden: Sequence[int] = (256, 256)
    latent: int = 128
    octaves: Tuple[int, int] = DEFAULT_OCTAVES
    feature_kind: str = "fourier_v1"
    probe_octaves: Tuple[int, int] = PROBE_OCTAVES
    a_y_anchors: Sequence[float] = DEFAULT_AXES
    a_z_anchors: Sequence[float] = DEFAULT_AXES
    feature_gate_init: str = "ones"

    @property
    def feature_dim(self):
        if self.feature_kind == PROBE_KIND:
            return feature_dim(PROBE_KIND, self.probe_octaves)
        if self.feature_kind == DENSE_KIND:
            return feature_dim(DENSE_KIND)
        return feature_dim("fourier_v1", self.octaves)

    @property
    def out_dim(self):
        return OUT_DIM

    def validate(self):
        if self.feature_kind == "fourier_v1":
            if tuple(self.octaves) != DEFAULT_OCTAVES:
                raise ValueError(
                    "artifact format v1 fixes features.kind=fourier_v1 with octaves [-3, 6]"
                )
            if self.feature_dim != 69:
                raise ValueError(f"feature_dim must be 69, got {self.feature_dim}")
        elif self.feature_kind == PROBE_KIND:
            if tuple(self.probe_octaves) != PROBE_OCTAVES:
                raise ValueError(f"{PROBE_KIND} fixes octaves {PROBE_OCTAVES}")
            if self.feature_dim != 231:
                raise ValueError(f"{PROBE_KIND} dim must be 231, got {self.feature_dim}")
        elif self.feature_kind == DENSE_KIND:
            if self.feature_dim != 1545:
                raise ValueError(f"{DENSE_KIND} dim must be 1545, got {self.feature_dim}")
        else:
            raise ValueError(f"unknown feature_kind {self.feature_kind!r}")
        if self.latent <= 0:
            raise ValueError("latent must be positive")
        for name, anchors in (("a_y", self.a_y_anchors), ("a_z", self.a_z_anchors)):
            a = np.asarray(anchors, dtype=np.float64)
            if a.ndim != 1 or len(a) < 2:
                raise ValueError(f"{name} anchors need at least two entries")
            if not np.all(np.diff(np.log10(a)) > 0):
                raise ValueError(f"{name} anchors must increase in log10")
            if a[0] < 0.1 - 1e-12 or a[-1] > 10.0 + 1e-12:
                raise ValueError(f"{name} anchors must stay inside [0.1, 10]")
        return self


def interp_indices_weights(a_y, a_z, a_y_anchors, a_z_anchors):
    """Bilinear cell + weights in ``log10(a)``; clamped at outer anchors.

    Returns ``(indices[..., 4], weights[..., 4])`` with order
    ``(iy0,iz0), (iy1,iz0), (iy0,iz1), (iy1,iz1)`` where the flat bank index is
    ``iz * n_y + iy``.
    """
    ay_a = np.asarray(a_y_anchors, dtype=np.float64)
    az_a = np.asarray(a_z_anchors, dtype=np.float64)
    ay = np.asarray(a_y, dtype=np.float64)
    az = np.asarray(a_z, dtype=np.float64)

    def axis_cell(a, anchors):
        n = len(anchors)
        lo, hi = np.log10(anchors[0]), np.log10(anchors[-1])
        t = (np.log10(a) - lo) / (hi - lo) * (n - 1)
        t = np.clip(t, 0.0, n - 1)
        i0 = np.floor(t).astype(np.int64)
        i0 = np.clip(i0, 0, n - 2) if n > 1 else np.zeros_like(i0)
        i1 = np.minimum(i0 + 1, n - 1)
        frac = t - i0
        return i0, i1, frac

    iy0, iy1, fy = axis_cell(ay, ay_a)
    iz0, iz1, fz = axis_cell(az, az_a)
    ny = len(ay_a)
    flat = np.stack(
        [
            iz0 * ny + iy0,
            iz0 * ny + iy1,
            iz1 * ny + iy0,
            iz1 * ny + iy1,
        ],
        axis=-1,
    )
    w = np.stack(
        [
            (1 - fy) * (1 - fz),
            fy * (1 - fz),
            (1 - fy) * fz,
            fy * fz,
        ],
        axis=-1,
    )
    return flat.astype(np.int64), w.astype(np.float64)


# --------------------------------------------------------------------------
# torch model
# --------------------------------------------------------------------------
try:  # torch is optional for pure-numpy-only consumers
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    _TORCH_ERROR = exc


if torch is not None:

    class Encoder(nn.Module):
        def __init__(self, cfg: ModelConfig):
            super().__init__()
            dims = [cfg.feature_dim, *list(cfg.hidden), cfg.latent]
            self.linears = nn.ModuleList(
                [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
            )
            for lin in self.linears:
                nn.init.kaiming_uniform_(lin.weight, a=5 ** 0.5)
                fan_in = lin.in_features
                bound = 1.0 / np.sqrt(fan_in)
                nn.init.uniform_(lin.bias, -bound, bound)
            # start near the (small) physical field scale instead of at O(1)
            nn.init.normal_(self.linears[-1].weight, std=1e-2)
            nn.init.zeros_(self.linears[-1].bias)
            self.register_buffer(
                "input_scale", torch.tensor(1.0 / feature_scale(cfg.feature_kind), dtype=torch.float64),
                persistent=False,
            )
            # Learnable per-feature gate.  It can suppress the high-frequency
            # fourier octaves that alias at large source distances, while the
            # exported first layer folds the gate in, so the artifact still
            # consumes the frozen fourier_v1 vector.
            gate = torch.ones(cfg.feature_dim, dtype=torch.float64)
            if getattr(cfg, "feature_gate_init", "ones") == "lowfreq":
                gate.zero_()
                gate[:9] = 1.0
                for axis in range(3):
                    base = 9 + axis * 20
                    gate[base:base + 10] = 1.0  # octaves -3..1, sin+cos
            self.feature_gate = nn.Parameter(gate)
            self.act = nn.SiLU()

        def forward(self, x):
            x = x * self.input_scale.to(dtype=x.dtype, device=x.device)
            x = x * self.feature_gate.to(dtype=x.dtype).unsqueeze(0)
            for i, lin in enumerate(self.linears):
                x = lin(x)
                if i != len(self.linears) - 1:
                    x = self.act(x)
            return x

        def layer_specs(self):
            specs = []
            for i, lin in enumerate(self.linears):
                specs.append(
                    {
                        "weight": f"enc.{i}.w",
                        "bias": f"enc.{i}.b",
                        "in": lin.in_features,
                        "out": lin.out_features,
                    }
                )
            return specs

    class DecoderBank(nn.Module):
        def __init__(self, cfg: ModelConfig):
            super().__init__()
            nz, ny = len(cfg.a_z_anchors), len(cfg.a_y_anchors)
            self.cfg = cfg
            # index = iz * n_y + iy, a_y fastest
            self.bank = nn.Parameter(torch.empty(nz * ny, cfg.latent, OUT_DIM))
            nn.init.normal_(self.bank, std=1.0 / np.sqrt(max(cfg.latent, 1)))
            self.register_buffer(
                "ay_anchors", torch.tensor(cfg.a_y_anchors, dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "az_anchors", torch.tensor(cfg.a_z_anchors, dtype=torch.float64),
                persistent=False,
            )

        def weights(self, a_y, a_z):
            idx, w = interp_indices_weights(
                a_y, a_z, self.cfg.a_y_anchors, self.cfg.a_z_anchors
            )
            idx = torch.as_tensor(idx, dtype=torch.long, device=self.bank.device)
            w = torch.as_tensor(w, dtype=self.bank.dtype, device=self.bank.device)
            wmat = (self.bank[idx] * w[:, :, None, None]).sum(dim=1)
            return wmat

    class NeuralBunchModel(nn.Module):
        """Additive per-source encoder followed by the interpolated decoder."""

        def __init__(self, cfg: ModelConfig):
            super().__init__()
            cfg.validate()
            self.cfg = cfg
            self.encoder = Encoder(cfg)
            self.decoder = DecoderBank(cfg)

        def forward(self, feats, src_w, tile_index, dec_idx, dec_w, n_tiles):
            """Batched forward with segment aggregation.

            feats: ``[N, 69]``; src_w: ``[N]`` signed ``q/e`` values (already
            ``-w``); tile_index: ``[N]`` tile ids; dec_idx/dec_w: ``[T, 4]``.
            Zero-weight rows are dropped before the encoder so invalid padded
            rows can never produce NaNs.
            """
            keep = src_w != 0
            if not bool(keep.any()):
                hw = feats.new_zeros((0, self.cfg.latent))
            else:
                feats = feats[keep]
                src_w = src_w[keep]
                tile_index = tile_index[keep]
                h = self.encoder(feats)
                hw = h * src_w.to(h.dtype).unsqueeze(-1)
            z = torch.zeros(
                (n_tiles, self.cfg.latent), dtype=hw.dtype, device=hw.device
            )
            if hw.shape[0]:
                z = z.index_add(0, tile_index, hw)
            # Accumulate the four bilinear bank corners one at a time.  This is
            # algebraically identical to gathering all four and summing, but
            # avoids a [T, 4, latent, out] temporary (400 MB at T=64).
            wmat = None
            for k in range(4):
                term = self.decoder.bank[dec_idx[:, k]] * dec_w[:, k, None, None]
                wmat = term if wmat is None else wmat + term
            return torch.einsum("tl,tlo->to", z, wmat)

        def forward_tile(self, feats, src_w, a_y, a_z):
            feats = torch.as_tensor(feats)
            src_w = torch.as_tensor(src_w, dtype=feats.dtype, device=feats.device)
            idx, w = interp_indices_weights(
                a_y, a_z, self.cfg.a_y_anchors, self.cfg.a_z_anchors
            )
            idx = torch.as_tensor(idx, dtype=torch.long, device=feats.device).view(1, 4)
            w = torch.as_tensor(w, dtype=feats.dtype, device=feats.device).view(1, 4)
            ti = torch.zeros(feats.shape[0], dtype=torch.long, device=feats.device)
            return self.forward(feats, src_w, ti, idx, w, 1)[0]

        @torch.no_grad()
        def artifact_tensors(self):
            tensors = []
            inv = (1.0 / feature_scale(self.cfg.feature_kind))[:, None]
            gate = self.encoder.feature_gate.detach().cpu().numpy()[:, None]
            for i, lin in enumerate(self.encoder.linears):
                w = lin.weight.detach().cpu().numpy().T
                if i == 0:
                    # bake the fixed input conditioning and the learned feature
                    # gate into the first layer
                    w = w * inv * gate
                tensors.append((f"enc.{i}.w", w))
                tensors.append((f"enc.{i}.b", lin.bias.detach().cpu().numpy()))
            bank = self.decoder.bank.detach().cpu().numpy()
            nz, ny = len(self.cfg.a_z_anchors), len(self.cfg.a_y_anchors)
            for iz in range(nz):
                for iy in range(ny):
                    tensors.append(
                        (f"dec.{iz}.{iy}.w", bank[iz * ny + iy].astype(np.float64))
                    )
            return tensors


# --------------------------------------------------------------------------
# pure-numpy forward from an exported artifact
# --------------------------------------------------------------------------
class NumpyArtifactModel:
    """Forward pass over the artifact tensor dictionary (float64 internally)."""

    def __init__(self, meta, tensors):
        import json

        self.meta = meta if isinstance(meta, dict) else json.loads(meta)
        self.t = {k: np.asarray(v, dtype=np.float64) for k, v in tensors.items()}
        self.layers = self.meta["encoder"]["layers"]
        self.latent = int(self.meta["encoder"]["output_dim"])
        dek = self.meta["decoder"]
        self.ay = np.asarray(dek["axes"]["a_y"], dtype=np.float64)
        self.az = np.asarray(dek["axes"]["a_z"], dtype=np.float64)
        self.bank_names = list(dek["bank"])

    def encode(self, feats):
        x = np.asarray(feats, dtype=np.float64)
        for i, spec in enumerate(self.layers):
            w = self.t[spec["weight"]]
            b = self.t[spec["bias"]]
            x = x @ w + b
            if i != len(self.layers) - 1:
                # SiLU
                x = x / (1.0 + np.exp(-x))
        return x

    def forward_tile(self, feats, src_w, a_y, a_z):
        feats = np.asarray(feats, dtype=np.float64)
        src_w = np.asarray(src_w, dtype=np.float64)
        keep = src_w != 0
        if keep.any():
            h = self.encode(feats[keep])
            z = np.sum(h * src_w[keep, None], axis=0)
        else:
            z = np.zeros(self.latent)
        idx, w = interp_indices_weights(a_y, a_z, self.ay, self.az)
        W = np.zeros((self.latent, OUT_DIM), dtype=np.float64)
        for k in range(4):
            W += w[k] * self.t[self.bank_names[int(idx[k])]]
        return W.T @ z
