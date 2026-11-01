#!/usr/bin/env python3
"""Dump the NumPy forward of a neural-bunch artifact for one tile.

Usage: probe_artifact_reference.py <artifact-prefix> <sources.json> <a_y> <a_z> <out.json>
sources.json: [{"r":[3],"u":[3],"w":weight}, ...]
The output contains the 1536 normalized tile values multiplied by the same
FieldNormalization(length=1) scale the C++ probe check uses, so both sides are
directly comparable.
"""
import json
import sys

import numpy as np

E = 1.602176634e-19
EPS0 = 8.8541878128e-12
C = 299792458.0
SCALE_E = E / (4.0 * np.pi * EPS0)
SCALE_B = SCALE_E / C


def load_tensors(meta, prefix):
    path = prefix.rsplit("/", 1)[0] + "/" + meta["artifact"]["file"] if "/" in prefix else meta["artifact"]["file"]
    with open(path, "rb") as stream:
        payload = stream.read()
    out = {}
    for spec in meta["tensors"]:
        offset = int(spec["offset_bytes"])
        count = int(spec["count"])
        values = np.frombuffer(payload, dtype="<f4", count=count, offset=offset).astype(np.float64)
        out[spec["name"]] = values.reshape([int(v) for v in spec["shape"]])
    return out


def point_field_dx(R, u):
    R = np.asarray(R, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    uu2 = np.sum(u * u, axis=-1)
    inv_g2 = 1.0 / (1.0 + uu2)
    speed = np.sqrt(uu2)
    static = speed == 0.0
    safe = np.where(static, 1.0, speed)
    bhat = u / safe[..., None]
    beta = u / np.sqrt(1.0 + uu2)[..., None]
    r_par = np.sum(R * bhat, axis=-1)
    r_perp = R - r_par[..., None] * bhat
    denom = (r_par * r_par + np.sum(r_perp * r_perp, axis=-1) * inv_g2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        E_mov = (inv_g2[..., None] * R) / denom[..., None]
        B_mov = np.cross(beta, E_mov)
        r2 = np.sum(R * R, axis=-1)
        E_st = R / (r2 ** 1.5)[..., None]
    return np.where(static[..., None], E_st, E_mov), np.where(static[..., None], 0.0, B_mov)


def features(meta, r, u, a_y, a_z):
    kind = meta["features"]["kind"]
    r = np.asarray(r, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    gamma = np.sqrt(1.0 + np.sum(u * u))
    base = [r[0], r[1], r[2], u[0], u[1], u[2], gamma, a_y, a_z]
    if kind == "fourier_v1":
        pieces = [np.array(base)]
        for axis in range(3):
            for octave in range(-3, 7):
                phase = np.pi * 2.0 ** octave * r[axis]
                pieces.append(np.array([np.sin(phase), np.cos(phase)]))
        return np.concatenate(pieces)
    offsets = np.asarray(meta["features"]["probe"]["cell_offsets"], dtype=np.float64)
    scale = np.array([1.0, a_y, a_z])
    probes = offsets * scale
    R = probes - r
    E, B = point_field_dx(R, np.broadcast_to(u, R.shape))
    eb = np.concatenate([E, B], axis=-1).reshape(-1)
    pieces = [np.array(base), eb]
    if kind == "fourier_probe_v1":
        terms = []
        for axis in range(3):
            for octave in range(-3, 2):
                phase = np.pi * 2.0 ** octave * r[axis]
                terms.extend([np.sin(phase), np.cos(phase)])
        pieces.append(np.array(terms))
    return np.concatenate(pieces)


def interp_weights(a, anchors):
    anchors = np.asarray(anchors, dtype=np.float64)
    n = len(anchors)
    t = (np.log10(a) - np.log10(anchors[0])) / (np.log10(anchors[-1]) - np.log10(anchors[0])) * (n - 1)
    t = np.clip(t, 0.0, n - 1)
    i0 = min(int(np.floor(t)), n - 2)
    i1 = min(i0 + 1, n - 1)
    return i0, i1, t - i0


def main():
    prefix, sources_path, a_y, a_z, out_path = sys.argv[1:6]
    a_y = float(a_y)
    a_z = float(a_z)
    with open(prefix + ".json") as stream:
        meta = json.load(stream)
    tensors = load_tensors(meta, prefix)
    sources = json.load(open(sources_path))
    feats = np.stack([features(meta, s["r"], s["u"], a_y, a_z) for s in sources])
    x = feats
    layers = meta["encoder"]["layers"]
    for i, layer in enumerate(layers):
        x = x @ tensors[layer["weight"]] + tensors[layer["bias"]]
        if i != len(layers) - 1:
            x = x / (1.0 + np.exp(-x))
    z = np.sum(x * (-np.array([s["w"] for s in sources]))[:, None], axis=0)
    iy0, iy1, fy = interp_weights(a_y, meta["decoder"]["axes"]["a_y"])
    iz0, iz1, fz = interp_weights(a_z, meta["decoder"]["axes"]["a_z"])
    ny = len(meta["decoder"]["axes"]["a_y"])
    names = meta["decoder"]["bank"]
    W = ((1 - fy) * (1 - fz) * tensors[names[iz0 * ny + iy0]]
         + fy * (1 - fz) * tensors[names[iz0 * ny + iy1]]
         + (1 - fy) * fz * tensors[names[iz1 * ny + iy0]]
         + fy * fz * tensors[names[iz1 * ny + iy1]]).reshape(meta["encoder"]["output_dim"], 1536)
    field = z @ W
    scale = np.array([SCALE_E if (i % 6) < 3 else SCALE_B for i in range(1536)])
    json.dump({"values": (field * scale).tolist(), "dim": int(meta["features"]["dim"])}, open(out_path, "w"))


if __name__ == "__main__":
    main()
