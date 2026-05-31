"""Compact Gaussian-splatting scene and renderer used by this benchmark.

The original paper evaluates large trained 3DGS scenes.  This module provides a
small deterministic 3D Gaussian scene with the same rendering primitives so the
LOD/chunking algorithm can be exercised without shipping multi-GB assets.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    f: float
    width: int
    height: int

    def as_dict(self) -> dict:
        return {
            "eye": self.eye.tolist(),
            "target": self.target.tolist(),
            "up": self.up.tolist(),
            "f": float(self.f),
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        return cls(
            eye=np.asarray(data["eye"], dtype=np.float32),
            target=np.asarray(data["target"], dtype=np.float32),
            up=np.asarray(data["up"], dtype=np.float32),
            f=float(data["f"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def camera_basis(cam: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _normalize(cam.target - cam.eye)
    right = _normalize(np.cross(forward, cam.up))
    up = _normalize(np.cross(right, forward))
    return right, up, forward


def generate_scene(seed: int = 7, fine_grid: int = 20, coarse_grid: int = 10) -> dict:
    rng = np.random.default_rng(seed)
    means = []
    sigma = []
    opacity = []
    colors = []

    # Three long facades with high-frequency window/stripe texture.  Dense base
    # splats make far regions expensive, matching the problem LODGE addresses.
    xs = np.linspace(-8.0, 8.0, fine_grid)
    zs = np.linspace(1.0, 7.0, coarse_grid)
    for wall_y, base_color, phase in [
        (5.0, np.array([0.62, 0.55, 0.47]), 0.0),
        (8.5, np.array([0.45, 0.58, 0.64]), 1.7),
        (-4.5, np.array([0.60, 0.47, 0.54]), 2.4),
    ]:
        for x in xs:
            for z in zs:
                means.append([x + rng.normal(0, 0.025), wall_y, z + rng.normal(0, 0.02)])
                pat = 0.5 + 0.5 * math.sin(2.5 * x + 2.0 * z + phase)
                colors.append(np.clip(base_color + 0.23 * pat + rng.normal(0, 0.025, 3), 0, 1))
                sigma.append(0.095 + 0.025 * rng.random())
                opacity.append(0.46 + 0.10 * rng.random())

    # Ground markers and foreground posts create close-view detail that should
    # remain high resolution while distant splats are smoothed/pruned.
    for x in np.linspace(-8, 8, fine_grid):
        for y in np.linspace(-5, 9, coarse_grid):
            means.append([x + rng.normal(0, 0.02), y + rng.normal(0, 0.02), 0.0])
            c = np.array([0.28, 0.33, 0.27]) + 0.12 * ((int((x + 8) * 2) + int((y + 5) * 2)) % 2)
            colors.append(np.clip(c + rng.normal(0, 0.015, 3), 0, 1))
            sigma.append(0.11 + 0.03 * rng.random())
            opacity.append(0.40 + 0.08 * rng.random())

    for x in np.linspace(-7.5, 7.5, 8):
        for z in np.linspace(0.2, 3.2, 6):
            means.append([x, 1.0 + 0.2 * math.sin(x), z])
            colors.append(np.array([0.12, 0.12, 0.12]) if int(z * 3) % 2 else np.array([0.9, 0.72, 0.18]))
            sigma.append(0.07)
            opacity.append(0.58)

    return {
        "means": np.asarray(means, dtype=np.float32),
        "sigma": np.asarray(sigma, dtype=np.float32),
        "opacity": np.asarray(opacity, dtype=np.float32),
        "colors": np.asarray(colors, dtype=np.float32),
    }


def make_cameras(width: int = 80, height: int = 54) -> tuple[list[Camera], list[Camera]]:
    train = []
    test = []
    for i, x in enumerate(np.linspace(-6.5, 6.5, 10)):
        eye = np.array([x, -7.0 + 0.7 * math.sin(i * 0.45), 2.1 + 0.2 * math.cos(i)], dtype=np.float32)
        target = np.array([x * 0.35, 2.8, 2.6], dtype=np.float32)
        train.append(Camera(eye, target, np.array([0, 0, 1], dtype=np.float32), 58.0, width, height))
    for i, x in enumerate(np.linspace(-6.0, 6.0, 5)):
        eye = np.array([x + 0.35, -6.4 + 0.55 * math.cos(i * 0.7), 2.2], dtype=np.float32)
        target = np.array([x * 0.30, 3.4, 2.5 + 0.2 * math.sin(i)], dtype=np.float32)
        test.append(Camera(eye, target, np.array([0, 0, 1], dtype=np.float32), 58.0, width, height))
    return train, test


def project_gaussians(scene: dict, cam: Camera, sigma_scale: float = 1.0) -> dict:
    right, up, forward = camera_basis(cam)
    rel = scene["means"] - cam.eye[None, :]
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    valid = z > 0.35
    u = cam.f * x / np.maximum(z, 1e-5) + cam.width * 0.5
    v = -cam.f * y / np.maximum(z, 1e-5) + cam.height * 0.5
    screen_sigma = cam.f * scene["sigma"] * sigma_scale / np.maximum(z, 1e-5)
    return {"u": u, "v": v, "z": z, "screen_sigma": screen_sigma, "valid": valid}


def render_scene(
    scene: dict,
    cam: Camera,
    indices: np.ndarray | None = None,
    opacity_scale: np.ndarray | float = 1.0,
    return_stats: bool = False,
) -> tuple[np.ndarray, dict] | np.ndarray:
    if indices is None:
        indices = np.arange(scene["means"].shape[0], dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    sub = {k: v[indices] for k, v in scene.items()}
    proj = project_gaussians(sub, cam)
    order = np.argsort(proj["z"])
    img = np.zeros((cam.height, cam.width, 3), dtype=np.float32)
    trans = np.ones((cam.height, cam.width), dtype=np.float32)
    importance = np.zeros(indices.shape[0], dtype=np.float32)
    visible = 0
    tile_counts: dict[tuple[int, int], int] = {}
    if np.isscalar(opacity_scale):
        opacity_vec = np.full(indices.shape[0], float(opacity_scale), dtype=np.float32)
    else:
        opacity_vec = np.asarray(opacity_scale, dtype=np.float32)

    for j in order:
        if not proj["valid"][j]:
            continue
        sig = float(max(proj["screen_sigma"][j], 0.45))
        rad = max(1, int(math.ceil(3.0 * sig)))
        u = float(proj["u"][j])
        v = float(proj["v"][j])
        x0 = max(0, int(math.floor(u - rad)))
        x1 = min(cam.width, int(math.ceil(u + rad + 1)))
        y0 = max(0, int(math.floor(v - rad)))
        y1 = min(cam.height, int(math.ceil(v + rad + 1)))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-0.5 * (((xx - u) ** 2 + (yy - v) ** 2) / (sig * sig))).astype(np.float32)
        alpha = np.clip(sub["opacity"][j] * opacity_vec[j] * g, 0.0, 0.98)
        contrib = trans[y0:y1, x0:x1] * alpha
        if contrib.max() <= 1e-5:
            continue
        img[y0:y1, x0:x1] += contrib[..., None] * sub["colors"][j]
        trans[y0:y1, x0:x1] *= 1.0 - alpha
        importance[j] = max(importance[j], float(contrib.max()))
        visible += 1
        tx0, tx1 = x0 // 16, (x1 - 1) // 16
        ty0, ty1 = y0 // 16, (y1 - 1) // 16
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile_counts[(tx, ty)] = tile_counts.get((tx, ty), 0) + 1

    img += trans[..., None] * np.array([0.02, 0.025, 0.03], dtype=np.float32)
    img = np.clip(img, 0, 1)
    if not return_stats:
        return img
    stats = {
        "importance": importance,
        "visible": visible,
        "tile_cost": float(np.mean(list(tile_counts.values()))) if tile_counts else 0.0,
        "max_tile_cost": float(max(tile_counts.values())) if tile_counts else 0.0,
        "loaded": int(indices.size),
    }
    return img, stats


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def simple_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    x = pred.reshape(-1, 3)
    y = gt.reshape(-1, 3)
    c1 = 0.01**2
    c2 = 0.03**2
    vals = []
    for ch in range(3):
        a = x[:, ch]
        b = y[:, ch]
        mux, muy = float(a.mean()), float(b.mean())
        vx, vy = float(a.var()), float(b.var())
        cov = float(((a - mux) * (b - muy)).mean())
        vals.append(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))
    return float(np.mean(vals))


def save_dataset(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    scene = generate_scene()
    reference_scene = generate_scene(fine_grid=24, coarse_grid=12)
    train, test = make_cameras()
    np.savez_compressed(path / "base_scene.npz", **scene)
    (path / "cameras.json").write_text(
        json.dumps({"train": [c.as_dict() for c in train], "test": [c.as_dict() for c in test]}, indent=2)
    )
    # Test images approximate real captures rather than the exact base model:
    # mostly the trained full representation, with deterministic residual detail
    # from a denser scene.  This keeps the full baseline finite and realistic.
    base_train = np.stack([render_scene(scene, c) for c in train])
    dense_train = np.stack([render_scene(reference_scene, c) for c in train])
    train_gt = np.clip(0.7 * base_train + 0.3 * dense_train, 0.0, 1.0)
    np.savez_compressed(path / "train_gt.npz", images=train_gt)

    base_test = np.stack([render_scene(scene, c) for c in test])
    dense_test = np.stack([render_scene(reference_scene, c) for c in test])
    test_gt = np.clip(0.7 * base_test + 0.3 * dense_test, 0.0, 1.0)
    np.savez_compressed(path / "test_gt.npz", images=test_gt)


def load_dataset(path: Path) -> tuple[dict, list[Camera], list[Camera], np.ndarray]:
    scene_np = np.load(path / "base_scene.npz")
    scene = {k: scene_np[k] for k in ["means", "sigma", "opacity", "colors"]}
    cams = json.loads((path / "cameras.json").read_text())
    train = [Camera.from_dict(x) for x in cams["train"]]
    test = [Camera.from_dict(x) for x in cams["test"]]
    gt = np.load(path / "test_gt.npz")["images"]
    return scene, train, test, gt


def load_train_targets(path: Path) -> np.ndarray:
    return np.load(path / "train_gt.npz")["images"]


def ensure_dataset(path: Path) -> None:
    required = ["base_scene.npz", "cameras.json", "train_gt.npz", "test_gt.npz"]
    if any(not (path / name).exists() for name in required):
        save_dataset(path)


def stack_renders(images: Iterable[np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(im, dtype=np.float32) for im in images], axis=0)
