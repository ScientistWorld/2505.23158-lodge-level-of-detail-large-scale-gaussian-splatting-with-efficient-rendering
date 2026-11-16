"""Hidden world generator for the LOD Gaussian-splatting benchmark.

Builds a long synthetic street canyon twice:

* the *true* scene: dense anisotropic surface Gaussians carrying a procedural
  facade/road/object texture.  Image targets are 3x3-supersampled renders of it.
* the *base* reconstruction: the same surfaces at half the density with
  box-averaged colours and small geometric jitter, then optimised on the
  training images (L1 + D-SSIM, no densification).  This plays the role of the
  trained full 3DGS model that LOD methods start from and is public.

Running ``python -m eval.test.world --build`` writes the public dataset to
``data/lodge_city`` and the hidden test cameras/targets to
``eval/test/hidden_set.npz`` (which also keeps the evaluator's own copy of the
trained base reconstruction for the fixed full-representation baseline).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

WORLD_SEED = 52305

FULL_CONFIG = {
    "name": "full",
    "x0": -6.0,
    "x1": 126.0,
    "half_width": 7.0,
    "true_spacing": 0.10,
    "base_spacing": 0.15,
    "focal": 96.0,
    "width": 160,
    "height": 96,
    "supersample": 3,
    "capture_x0": 2.0,
    "capture_x1": 117.0,
    "capture_step": 1.25,
    "fly_test": [25.0, 65.0, 0.5, 1.3],
    "fly_val": [70.0, 100.0, 0.5, -1.3],
    "off_path_views": 30,
    "posts_every": 15.0,
    "n_cars": 14,
    "base_train_steps": 5000,
}

SMOKE_CONFIG = dict(FULL_CONFIG)
SMOKE_CONFIG.update(
    {
        "name": "smoke",
        "x0": -4.0,
        "x1": 34.0,
        "true_spacing": 0.2,
        "base_spacing": 0.3,
        "supersample": 2,
        "capture_x0": 2.0,
        "capture_x1": 28.0,
        "capture_step": 2.0,
        "fly_test": [8.0, 16.0, 0.5, 1.3],
        "fly_val": [18.0, 26.0, 0.5, -1.3],
        "off_path_views": 6,
        "n_cars": 3,
        "base_train_steps": 100,
    }
)


# --------------------------------------------------------------------------
# Hash noise and colour helpers
# --------------------------------------------------------------------------


def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    h = (ix.astype(np.int64).astype(np.uint64) * np.uint64(73856093)) ^ (
        iy.astype(np.int64).astype(np.uint64) * np.uint64(19349663)
    ) ^ np.uint64(seed * 83492791 % (2**31))
    h = h ^ (h >> np.uint64(13))
    h = h * np.uint64(1274126177)
    h = h ^ (h >> np.uint64(16))
    return (h & np.uint64(0xFFFF)).astype(np.float64) / 65536.0


def _cell_noise(a: np.ndarray, b: np.ndarray, cell: float, seed: int) -> np.ndarray:
    return _hash2(np.floor(a / cell), np.floor(b / cell), seed)


PALETTE = np.array(
    [
        [0.72, 0.62, 0.50], [0.62, 0.40, 0.33], [0.78, 0.74, 0.66], [0.55, 0.58, 0.62],
        [0.80, 0.68, 0.45], [0.48, 0.52, 0.45], [0.70, 0.50, 0.42], [0.86, 0.83, 0.78],
        [0.58, 0.46, 0.55], [0.66, 0.70, 0.74],
    ]
)
CAR_COLORS = np.array(
    [[0.75, 0.10, 0.10], [0.12, 0.22, 0.55], [0.85, 0.85, 0.82], [0.15, 0.15, 0.16], [0.55, 0.57, 0.60], [0.90, 0.72, 0.15], [0.20, 0.45, 0.30]]
)


# --------------------------------------------------------------------------
# Procedural textures (3D point -> RGB) for each surface family
# --------------------------------------------------------------------------


def make_buildings(rng: np.random.Generator, s0: float, s1: float, max_h: float = 12.5) -> list[dict]:
    out = []
    s = s0
    while s < s1:
        w = float(rng.uniform(8.0, 20.0))
        b = {
            "s0": s,
            "s1": min(s + w, s1),
            "height": float(rng.uniform(8.0, max_h)),
            "base": np.clip(PALETTE[rng.integers(len(PALETTE))] * rng.uniform(0.88, 1.08), 0, 1),
            "floor_h": float(rng.uniform(3.0, 3.6)),
            "win_w": float(rng.uniform(0.8, 1.4)),
            "win_h": float(rng.uniform(1.1, 1.7)),
            "win_sp": float(rng.uniform(1.9, 2.9)),
            "win_col": np.array([0.16, 0.20, 0.26]) * rng.uniform(0.8, 1.3),
            "stripe_p": float(rng.uniform(0.22, 0.36)),
            "stripe_a": float(rng.uniform(0.03, 0.08)),
            "cell": float(rng.uniform(0.3, 0.6)),
            "sign": np.clip(rng.uniform(0.1, 0.95, 3), 0, 1),
            "seed": int(rng.integers(1, 10**6)),
        }
        out.append(b)
        s += w
    return out


def facade_color(s: np.ndarray, z: np.ndarray, buildings: list[dict]) -> np.ndarray:
    col = np.zeros(s.shape + (3,))
    for b in buildings:
        m = (s >= b["s0"]) & (s < b["s1"])
        if not np.any(m):
            continue
        ss, zz = s[m], z[m]
        c = np.repeat(b["base"][None, :], ss.size, axis=0)
        stripe = np.sign(np.sin(2 * math.pi * zz / b["stripe_p"]))
        c = c * (1.0 + b["stripe_a"] * stripe)[:, None]
        n = _cell_noise(ss, zz, b["cell"], b["seed"])
        c = c * (0.92 + 0.16 * n)[:, None]
        zf = np.mod(zz, b["floor_h"])
        cornice = (zf < 0.18) & (zz > 1.0)
        c[cornice] = np.clip(c[cornice] * 1.25 + 0.05, 0, 1)
        u = np.mod(ss - b["s0"], b["win_sp"])
        win = (np.abs(u - 0.5 * b["win_sp"]) < 0.5 * b["win_w"]) & (zf > 0.7) & (zf < 0.7 + b["win_h"]) & (zz > b["floor_h"])
        refl = 0.12 * (zf - 0.7) / b["win_h"]
        wc = b["win_col"][None, :] + refl[:, None] + 0.05 * _cell_noise(ss, zz, 0.15, b["seed"] + 7)[:, None]
        c[win] = wc[win]
        u2 = np.mod(ss - b["s0"], 2 * b["win_sp"])
        shop = (zz < b["floor_h"]) & (zz > 0.35) & (zz < 2.5) & (u2 > 0.35) & (u2 < 2 * b["win_sp"] - 0.35)
        c[shop] = np.array([0.22, 0.27, 0.31]) + 0.18 * ((zz[shop] - 0.35) / 2.15)[:, None]
        sign = (zz >= 2.65) & (zz < 2.95)
        c[sign] = b["sign"]
        edge = (ss - b["s0"] < 0.12) | (b["s1"] - ss < 0.12)
        c[edge] = c[edge] * 0.6
        top = (b["height"] - zz) < 0.25
        c[top] = np.clip(c[top] * 0.75, 0, 1)
        col[m] = c
    return np.clip(col, 0, 1)


def facade_height(s: np.ndarray, buildings: list[dict]) -> np.ndarray:
    h = np.zeros_like(s)
    for b in buildings:
        m = (s >= b["s0"]) & (s < b["s1"])
        h[m] = b["height"]
    return h


def ground_color(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ay = np.abs(y)
    asphalt = 0.24 + 0.06 * _cell_noise(x, y, 0.35, 11) - 0.03 * _cell_noise(x, y, 1.7, 12)
    col = np.stack([asphalt, asphalt, asphalt * 1.04], axis=-1)
    dash = (ay < 0.08) & (np.mod(x, 6.0) < 3.0)
    edge = np.abs(ay - 4.2) < 0.07
    col[dash | edge] = np.array([0.88, 0.87, 0.80])
    curb = (ay >= 4.4) & (ay < 4.65)
    col[curb] = np.array([0.62, 0.62, 0.60])
    side = ay >= 4.65
    tile = ((np.floor(x / 0.6) + np.floor(y / 0.6)) % 2).astype(np.float64)
    joint = (np.mod(x, 0.6) < 0.035) | (np.mod(y, 0.6) < 0.035)
    sv = 0.50 + 0.06 * tile + 0.04 * _cell_noise(x, y, 0.6, 13) - 0.18 * joint
    col[side] = np.stack([sv, sv * 0.98, sv * 0.94], axis=-1)[side]
    return np.clip(col, 0, 1)


# --------------------------------------------------------------------------
# Surface sampling
# --------------------------------------------------------------------------


def frame_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.where(np.abs(n[:, 2:3]) > 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]))
    t1 = np.cross(a, n)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    return t1, t2


def rotmat_to_quat(r: np.ndarray) -> np.ndarray:
    m00, m01, m02 = r[:, 0, 0], r[:, 0, 1], r[:, 0, 2]
    m10, m11, m12 = r[:, 1, 0], r[:, 1, 1], r[:, 1, 2]
    m20, m21, m22 = r[:, 2, 0], r[:, 2, 1], r[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.zeros((r.shape[0], 4))
    c0 = tr > 0
    c1 = ~c0 & (m00 > m11) & (m00 > m22)
    c2 = ~c0 & ~c1 & (m11 > m22)
    c3 = ~c0 & ~c1 & ~c2
    s = np.sqrt(np.maximum(tr + 1.0, 1e-12)) * 2
    q[c0] = np.stack([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s], -1)[c0]
    s = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 1e-12)) * 2
    q[c1] = np.stack([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s], -1)[c1]
    s = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 1e-12)) * 2
    q[c2] = np.stack([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s], -1)[c2]
    s = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 1e-12)) * 2
    q[c3] = np.stack([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s], -1)[c3]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


class Surfaces:
    """Accumulates surface samples: position, normal, and a colour function id."""

    def __init__(self):
        self.parts = []

    def add(self, pts: np.ndarray, normals: np.ndarray, color_fn) -> None:
        if pts.shape[0]:
            self.parts.append((pts, normals, color_fn))


def plane_points(origin, u_dir, v_dir, u_len, v_len, h):
    nu = max(1, int(round(u_len / h)))
    nv = max(1, int(round(v_len / h)))
    hu, hv = u_len / nu, v_len / nv
    iu, iv = np.meshgrid((np.arange(nu) + 0.5) * hu, (np.arange(nv) + 0.5) * hv, indexing="ij")
    pts = np.asarray(origin)[None, :] + iu.reshape(-1, 1) * np.asarray(u_dir)[None, :] + iv.reshape(-1, 1) * np.asarray(v_dir)[None, :]
    n = np.cross(u_dir, v_dir).astype(np.float64)
    n /= np.linalg.norm(n)
    return pts, np.repeat(n[None, :], pts.shape[0], axis=0)


def box_points(center, size, h):
    cx, cy, cz = center
    sx, sy, sz = size
    pts, nrm = [], []
    faces = [
        ((cx - sx / 2, cy - sy / 2, cz - sz / 2), (1, 0, 0), (0, 0, 1), sx, sz),  # -y face
        ((cx - sx / 2, cy + sy / 2, cz - sz / 2), (0, 0, 1), (1, 0, 0), sz, sx),  # +y face
        ((cx - sx / 2, cy - sy / 2, cz - sz / 2), (0, 0, 1), (0, 1, 0), sz, sy),  # -x face
        ((cx + sx / 2, cy - sy / 2, cz - sz / 2), (0, 1, 0), (0, 0, 1), sy, sz),  # +x face
        ((cx - sx / 2, cy - sy / 2, cz + sz / 2), (1, 0, 0), (0, 1, 0), sx, sy),  # top
    ]
    for o, u, v, lu, lv in faces:
        p, n = plane_points(np.array(o, dtype=np.float64), np.array(u, dtype=np.float64), np.array(v, dtype=np.float64), lu, lv, h)
        pts.append(p)
        nrm.append(n)
    return np.concatenate(pts), np.concatenate(nrm)


def sphere_points(center, radius, h):
    n = max(8, int(math.ceil(4 * math.pi * radius * radius / (h * h))))
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    d = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], -1)
    return np.asarray(center)[None, :] + radius * d, d


def build_surfaces(cfg: dict, h: float) -> Surfaces:
    rng = np.random.default_rng(WORLD_SEED)
    x0, x1, hw = cfg["x0"], cfg["x1"], cfg["half_width"]
    left = make_buildings(rng, x0, x1)
    right = make_buildings(rng, x0, x1)
    end_a = make_buildings(rng, -hw, hw)
    end_b = make_buildings(rng, -hw, hw)
    surf = Surfaces()
    max_h = 12.5

    # Facades at y = +hw (left when looking along +x) and y = -hw.
    for y_wall, blds in ((hw, left), (-hw, right)):
        p, n = plane_points(np.array([x0, y_wall, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), x1 - x0, max_h, h)
        keep = p[:, 2] < facade_height(p[:, 0], blds)
        surf.add(p[keep], n[keep], ("facade_x", blds))
    # End walls closing the street.
    for x_wall, blds in ((x0, end_a), (x1, end_b)):
        p, n = plane_points(np.array([x_wall, -hw, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), 2 * hw, max_h, h)
        keep = p[:, 2] < np.maximum(facade_height(p[:, 1], blds), 9.0)
        surf.add(p[keep], n[keep], ("facade_y", blds))
    # Ground.
    p, n = plane_points(np.array([x0, -hw, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), x1 - x0, 2 * hw, h)
    surf.add(p, n, ("ground", None))

    # Lamp posts.
    xp = cfg["capture_x0"] + 1.0
    while xp < x1 - 2:
        for side in (-1.0, 1.0):
            p, n = box_points((xp, side * 4.9, 2.5), (0.16, 0.16, 5.0), h * 0.6)
            surf.add(p, n, ("solid", np.array([0.18, 0.19, 0.21])))
            p, n = box_points((xp, side * 4.55, 5.05), (0.25, 0.7, 0.12), h * 0.6)
            surf.add(p, n, ("solid", np.array([0.95, 0.90, 0.70])))
        xp += cfg["posts_every"]
    # Trees between posts.
    xt = cfg["capture_x0"] + 1.0 + 0.5 * cfg["posts_every"]
    k = 0
    while xt < x1 - 2:
        side = 1.0 if k % 2 == 0 else -1.0
        p, n = box_points((xt, side * 6.1, 1.3), (0.3, 0.3, 2.6), h)
        surf.add(p, n, ("solid", np.array([0.35, 0.25, 0.18])))
        p, n = sphere_points(np.array([xt, side * 6.1, 3.7]), 1.3, h)
        surf.add(p, n, ("foliage", int(rng.integers(1, 10**6))))
        xt += cfg["posts_every"]
        k += 1
    # Parked cars.
    xs = np.sort(rng.uniform(cfg["capture_x0"], x1 - 6, size=cfg["n_cars"]))
    last = -1e9
    for xc in xs:
        if xc - last < 5.5:
            continue
        last = xc
        side = 1.0 if rng.random() < 0.5 else -1.0
        col = CAR_COLORS[rng.integers(len(CAR_COLORS))]
        p, n = box_points((xc, side * 3.4, 0.68), (4.2, 1.8, 0.86), h)
        surf.add(p, n, ("solid", col))
        p, n = box_points((xc - 0.2, side * 3.4, 1.33), (2.4, 1.6, 0.44), h)
        surf.add(p, n, ("car_cabin", col))
    return surf


def evaluate_color(pts: np.ndarray, kind) -> np.ndarray:
    name, payload = kind
    if name == "facade_x":
        return facade_color(pts[:, 0], pts[:, 2], payload)
    if name == "facade_y":
        return facade_color(pts[:, 1], pts[:, 2], payload)
    if name == "ground":
        return ground_color(pts[:, 0], pts[:, 1])
    if name == "solid":
        n = _cell_noise(pts[:, 0] + pts[:, 1], pts[:, 2], 0.2, 5)
        return np.clip(payload[None, :] * (0.9 + 0.2 * n)[:, None], 0, 1)
    if name == "car_cabin":
        dark = np.array([0.10, 0.12, 0.16])
        side = (pts[:, 2] < 1.52) & (pts[:, 2] > 1.15)
        c = np.repeat(payload[None, :], pts.shape[0], axis=0)
        c[side] = dark + 0.1 * (pts[side, 2:3] - 1.15)
        return np.clip(c, 0, 1)
    if name == "foliage":
        n1 = _cell_noise(pts[:, 0] * 1.7 + pts[:, 2], pts[:, 1] + pts[:, 2] * 1.3, 0.25, payload)
        n2 = _cell_noise(pts[:, 0], pts[:, 2], 0.6, payload + 1)
        g = np.stack([0.16 + 0.10 * n1, 0.36 + 0.16 * n1 - 0.05 * n2, 0.12 + 0.06 * n2], -1)
        return np.clip(g, 0, 1)
    raise ValueError(name)


def build_scene(cfg: dict, kind: str) -> dict:
    """Return the true (``kind='true'``) or base (``kind='base'``) Gaussian scene."""
    h = cfg["true_spacing"] if kind == "true" else cfg["base_spacing"]
    surf = build_surfaces(cfg, h)
    rng = np.random.default_rng(WORLD_SEED + (1 if kind == "true" else 2))
    means, scales, quats, opac, colors = [], [], [], [], []
    for pts, nrm, ckind in surf.parts:
        t1, t2 = frame_from_normal(nrm)
        if kind == "true":
            col = evaluate_color(pts, ckind)
            st = np.full(pts.shape[0], 0.6 * h)
            op = np.full(pts.shape[0], 0.97)
            p = pts
        else:
            offs = (np.arange(4) + 0.5) / 4 - 0.5
            acc = np.zeros((pts.shape[0], 3))
            for oa in offs:
                for ob in offs:
                    acc += evaluate_color(pts + (oa * h) * t1 + (ob * h) * t2, ckind)
            col = acc / 16.0
            st = 0.6 * h * rng.uniform(0.9, 1.1, pts.shape[0])
            op = rng.uniform(0.88, 0.99, pts.shape[0])
            p = pts + rng.normal(0, 0.02, (pts.shape[0], 1)) * t1 + rng.normal(0, 0.02, (pts.shape[0], 1)) * t2 + rng.normal(0, 0.01, (pts.shape[0], 1)) * nrm
        rot = np.stack([t1, t2, nrm], axis=-1)
        means.append(p)
        scales.append(np.stack([st, st, 0.15 * st], -1))
        quats.append(rotmat_to_quat(rot))
        opac.append(op)
        colors.append(col)
    return {
        "means": np.concatenate(means).astype(np.float32),
        "scales": np.concatenate(scales).astype(np.float32),
        "quats": np.concatenate(quats).astype(np.float32),
        "opacities": np.concatenate(opac).astype(np.float32),
        "colors": np.clip(np.concatenate(colors), 0, 1).astype(np.float32),
    }


# --------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------


def _cam(eye, yaw, pitch) -> dict:
    return {"eye": [float(e) for e in eye], "yaw": float(yaw), "pitch": float(pitch)}


def camera_sets(cfg: dict) -> dict:
    deg = math.pi / 180.0
    capture = []
    x = cfg["capture_x0"]
    while x <= cfg["capture_x1"] + 1e-6:
        eye = (x, 0.35 * math.sin(x / 9.0), 1.7 + 0.05 * math.sin(x / 3.0))
        for off in (0.0, 55.0, -55.0):
            capture.append(_cam(eye, (off + 2.0 * math.sin(x / 5.0)) * deg, 3.0 * deg))
        x += cfg["capture_step"]
    sets = {"train": [], "val_on_path": [], "test_on_path": []}
    for i, c in enumerate(capture):
        if i % 8 == 0:
            sets["test_on_path"].append(c)
        elif i % 8 == 4:
            sets["val_on_path"].append(c)
        else:
            sets["train"].append(c)

    def fly(spec, phase):
        a, b, step, y = spec
        out = []
        n = int(round((b - a) / step)) + 1
        for k in range(n):
            xx = a + k * step
            out.append(_cam((xx, y, 1.75), 6.0 * deg * math.sin(xx / 6.0 + phase), 2.0 * deg))
        return out

    sets["test_flythrough"] = fly(cfg["fly_test"], 0.0)
    sets["val_flythrough"] = fly(cfg["fly_val"], 1.0)

    def off_path(seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(cfg["off_path_views"]):
            eye = (rng.uniform(cfg["capture_x0"] + 6, cfg["capture_x1"] - 5), rng.uniform(-2.2, 2.2), rng.uniform(1.2, 4.5))
            out.append(_cam(eye, rng.uniform(-math.pi, math.pi), rng.uniform(-10.0, 15.0) * deg))
        return out

    sets["test_off_path"] = off_path(WORLD_SEED + 101)
    sets["val_off_path"] = off_path(WORLD_SEED + 202)
    return sets


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def train_base(scene: dict, cams: list, images: np.ndarray, steps: int, device) -> dict:
    """Optimise the base reconstruction on the training images (3DGS-style, no densification).

    This makes the public full representation a converged model of the
    training views, as in the paper, instead of a raw geometric downsample.
    """
    import torch
    import torch.nn.functional as F

    from eval.test.render import render_gaussians

    torch.manual_seed(0)
    rng = np.random.default_rng(WORLD_SEED + 3)
    t = {k: torch.as_tensor(v, device=device) for k, v in scene.items()}
    means = torch.nn.Parameter(t["means"].clone())
    log_s = torch.nn.Parameter(torch.log(t["scales"]))
    quats = torch.nn.Parameter(t["quats"].clone())
    op = torch.nn.Parameter(torch.logit(t["opacities"].clamp(1e-4, 1 - 1e-4)))
    col = torch.nn.Parameter(torch.logit(t["colors"].clamp(1e-3, 1 - 1e-3)))
    opt = torch.optim.Adam(
        [{"params": [means], "lr": 1e-4}, {"params": [log_s], "lr": 5e-3}, {"params": [quats], "lr": 1e-3},
         {"params": [op], "lr": 2.5e-2}, {"params": [col], "lr": 5e-3}], eps=1e-15)
    target = torch.as_tensor(images, device=device).float() / 255.0
    x = torch.arange(11, dtype=torch.float32, device=device) - 5.0
    g = torch.exp(-(x * x) / 4.5)
    g = g / g.sum()
    win = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()

    def dssim(a, b):
        a = a.permute(2, 0, 1)[None]
        b = b.permute(2, 0, 1)[None]
        ma, mb = F.conv2d(a, win, padding=5, groups=3), F.conv2d(b, win, padding=5, groups=3)
        saa = F.conv2d(a * a, win, padding=5, groups=3) - ma * ma
        sbb = F.conv2d(b * b, win, padding=5, groups=3) - mb * mb
        sab = F.conv2d(a * b, win, padding=5, groups=3) - ma * mb
        s = ((2 * ma * mb + 1e-4) * (2 * sab + 9e-4)) / ((ma * ma + mb * mb + 1e-4) * (saa + sbb + 9e-4))
        return 1.0 - s.mean()

    for step in range(steps):
        i = int(rng.integers(len(cams)))
        out = render_gaussians(means, torch.exp(log_s), quats, torch.sigmoid(op), torch.sigmoid(col), cams[i])
        loss = 0.8 * torch.abs(out["image"] - target[i]).mean() + 0.2 * dssim(out["image"], target[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"[world] base training step {step}: loss {float(loss):.4f}", flush=True)
    return {
        "means": means.detach().cpu().numpy().astype(np.float32),
        "scales": torch.exp(log_s).detach().cpu().numpy().astype(np.float32),
        "quats": quats.detach().cpu().numpy().astype(np.float32),
        "opacities": torch.sigmoid(op).detach().cpu().numpy().astype(np.float32),
        "colors": torch.sigmoid(col).detach().cpu().numpy().astype(np.float32),
    }


def build(cfg: dict, data_dir: Path, hidden_path: Path) -> None:
    import torch

    from eval.test.render import Camera, default_device, render_supersampled, to_uint8

    device = default_device()
    print(f"[world] config={cfg['name']} device={device}", flush=True)
    true_scene = build_scene(cfg, "true")
    base_scene = build_scene(cfg, "base")
    print(f"[world] true Gaussians={true_scene['means'].shape[0]} base Gaussians={base_scene['means'].shape[0]}", flush=True)
    sets = camera_sets(cfg)
    gt = {k: torch.as_tensor(v, device=device) for k, v in true_scene.items()}

    def render_set(cams: list[dict]) -> np.ndarray:
        out = []
        for c in cams:
            cam = Camera(np.asarray(c["eye"]), c["yaw"], c["pitch"], cfg["focal"], cfg["width"], cfg["height"])
            with torch.no_grad():
                out.append(to_uint8(render_supersampled(gt, cam, cfg["supersample"])))
        return np.stack(out)

    images = {}
    for key, cams in sets.items():
        images[key] = render_set(cams)
        print(f"[world] rendered {key}: {len(cams)} views", flush=True)
    del gt
    train_cams = [Camera(np.asarray(c["eye"]), c["yaw"], c["pitch"], cfg["focal"], cfg["width"], cfg["height"]) for c in sets["train"]]
    from eval.test.render import psnr_u8, render_gaussians

    def val_psnr(scene: dict) -> float:
        t = {k: torch.as_tensor(v, device=device) for k, v in scene.items()}
        vals = []
        for c, img in zip(sets["val_on_path"], images["val_on_path"]):
            cam = Camera(np.asarray(c["eye"]), c["yaw"], c["pitch"], cfg["focal"], cfg["width"], cfg["height"])
            with torch.no_grad():
                out = render_gaussians(t["means"], t["scales"], t["quats"], t["opacities"], t["colors"], cam)
            vals.append(psnr_u8(to_uint8(out["image"]), img))
        return float(np.mean(vals))

    before = val_psnr(base_scene)
    base_scene = train_base(base_scene, train_cams, images["train"], cfg["base_train_steps"], device)
    print(f"[world] base reconstruction val PSNR: init {before:.3f} dB -> trained {val_psnr(base_scene):.3f} dB", flush=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / "base_gaussians.npz", **base_scene)
    public_sets = {k: v for k, v in sets.items() if not k.startswith("test_")}
    meta = {
        "intrinsics": {"f": cfg["focal"], "width": cfg["width"], "height": cfg["height"]},
        "sets": public_sets,
        "notes": "train: optimisation views with targets; val_*: public validation views mirroring the hidden test conditions.",
    }
    (data_dir / "cameras.json").write_text(json.dumps(meta, indent=1))
    np.savez_compressed(data_dir / "images.npz", **{k: v for k, v in images.items() if not k.startswith("test_")})

    hidden = {"world_config": np.array(json.dumps(cfg))}
    for k, v in base_scene.items():
        hidden[f"full_{k}"] = v
    for key in ("test_on_path", "test_flythrough", "test_off_path"):
        cams = sets[key]
        hidden[f"{key}_eye"] = np.array([c["eye"] for c in cams], dtype=np.float64)
        hidden[f"{key}_yaw"] = np.array([c["yaw"] for c in cams], dtype=np.float64)
        hidden[f"{key}_pitch"] = np.array([c["pitch"] for c in cams], dtype=np.float64)
        hidden[f"{key}_images"] = images[key]
    hidden_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(hidden_path, **hidden)
    print(f"[world] wrote {data_dir} and {hidden_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data/lodge_city"))
    parser.add_argument("--hidden", type=Path, default=Path("eval/test/hidden_set.npz"))
    args = parser.parse_args()
    cfg = SMOKE_CONFIG if args.smoke else FULL_CONFIG
    if args.build:
        build(cfg, args.data_dir, args.hidden)


if __name__ == "__main__":
    main()
