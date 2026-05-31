"""Compact implementation of the LODGE Gaussian-splatting algorithm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.gaussian_scene import ensure_dataset, load_dataset, render_scene, stack_renders


def copy_scene(scene: dict) -> dict:
    return {k: np.array(v, copy=True) for k, v in scene.items()}


def smooth_scene(scene: dict, depth: float, focal: float, smooth_s: float) -> dict:
    out = copy_scene(scene)
    # Eq. (3) in the paper adds a depth-aware 3D smoothing filter.  In this
    # isotropic compact renderer sigma is the Gaussian radius, so adding the
    # filter covariance corresponds to quadrature addition of radii.
    filter_radius = smooth_s * depth / focal
    base_sigma = out["sigma"]
    out["sigma"] = np.sqrt(base_sigma * base_sigma + filter_radius * filter_radius).astype(np.float32)
    # Preserve integrated density when smoothing by attenuating opacity in
    # proportion to the original/filter covariance determinant ratio.
    attenuation = (base_sigma / np.maximum(out["sigma"], 1e-6)) ** 1.5
    out["opacity"] = np.clip(out["opacity"] * attenuation.astype(np.float32), 0.0, 0.98)
    return out


def importance_scores(scene: dict, cameras: list, indices: np.ndarray | None = None) -> np.ndarray:
    if indices is None:
        indices = np.arange(scene["means"].shape[0])
    scores = np.zeros(indices.shape[0], dtype=np.float32)
    for cam in cameras:
        _, stats = render_scene(scene, cam, indices=indices, return_stats=True)
        scores = np.maximum(scores, stats["importance"])
    return scores


def prune_by_importance(scene: dict, cameras: list, threshold: float) -> dict:
    scores = importance_scores(scene, cameras)
    keep = scores >= threshold
    if keep.sum() < max(8, int(0.05 * keep.size)):
        cutoff = np.quantile(scores, 0.65)
        keep = scores >= cutoff
    return {k: v[keep] for k, v in scene.items()}


def pseudo_finetune(scene: dict, cameras: list, steps: int, seed: int) -> dict:
    """Lightweight deterministic stand-in for the paper's 1000-step LOD finetune.

    The paper optimizes only the current LOD set after each pruning round.  For
    this compact CPU benchmark we avoid expensive gradient descent but keep the
    same procedural stage: update only surviving LOD parameters using visibility
    statistics from training cameras.
    """

    rng = np.random.default_rng(seed)
    out = copy_scene(scene)
    scores = importance_scores(out, cameras)
    if scores.max() > 0:
        norm = scores / scores.max()
        out["opacity"] = np.clip(out["opacity"] * (0.94 + 0.10 * norm), 0.02, 0.98).astype(np.float32)
    jitter = rng.normal(0.0, 0.002 / max(1, steps), out["colors"].shape).astype(np.float32)
    out["colors"] = np.clip(out["colors"] + jitter, 0.0, 1.0)
    return out


def build_lod_levels(scene: dict, cameras: list, thresholds: list[float], gamma: float, smooth_s: float) -> list[dict]:
    levels = [copy_scene(scene)]
    focal = float(np.mean([c.f for c in cameras]))
    for level_idx, depth in enumerate(thresholds, start=1):
        lod = smooth_scene(scene, depth, focal, smooth_s)
        for round_idx, mult in enumerate([0.2, 0.6, 1.0]):
            lod = prune_by_importance(lod, cameras, gamma * mult)
            lod = pseudo_finetune(lod, cameras, steps=1000, seed=level_idx * 100 + round_idx)
        levels.append(lod)
    return levels


def active_indices_for_center(levels: list[dict], thresholds: list[float], center: np.ndarray, radius_offset: float = 0.0) -> list[np.ndarray]:
    d = [0.0] + [max(0.0, t - radius_offset) for t in thresholds] + [np.inf]
    selected = []
    for lvl, scene in enumerate(levels):
        dist = np.linalg.norm(scene["means"] - center[None, :], axis=1)
        selected.append(np.where((dist >= d[lvl]) & (dist < d[lvl + 1]))[0])
    return selected


def gather_active_scene(levels: list[dict], active: list[np.ndarray]) -> dict:
    parts = []
    for scene, inds in zip(levels, active):
        if len(inds) == 0:
            continue
        parts.append({k: v[inds] for k, v in scene.items()})
    if not parts:
        return {k: v[:0] for k, v in levels[0].items()}
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in levels[0].keys()}


def render_lod_at_camera(levels: list[dict], thresholds: list[float], cam) -> tuple[np.ndarray, dict]:
    active = active_indices_for_center(levels, thresholds, cam.eye, 0.0)
    active_scene = gather_active_scene(levels, active)
    img, stats = render_scene(active_scene, cam, return_stats=True)
    return img, {"loaded": stats["loaded"], "visible": stats["visible"], "tile_cost": stats["tile_cost"]}


def threshold_cost(scene: dict, cameras: list, thresholds: list[float], gamma: float, smooth_s: float) -> float:
    levels = build_lod_levels(scene, cameras[:3], thresholds, gamma=gamma, smooth_s=smooth_s)
    costs = []
    for cam in cameras[:3]:
        _, stats = render_lod_at_camera(levels, thresholds, cam)
        costs.append(stats["tile_cost"])
    return float(np.mean(costs))


def select_thresholds(scene: dict, cameras: list, n_levels: int, candidates: list[float], gamma: float, smooth_s: float) -> list[float]:
    selected: list[float] = []
    for _ in range(n_levels):
        best = None
        for cand in candidates:
            if selected and cand <= selected[-1] + 0.5:
                continue
            trial = selected + [cand]
            cost = threshold_cost(scene, cameras, trial, gamma=gamma, smooth_s=smooth_s)
            if best is None or cost < best[0]:
                best = (cost, cand)
        if best is None:
            break
        selected.append(float(best[1]))
    return selected


def kmeans(points: np.ndarray, k: int, seed: int = 3, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = points[rng.choice(points.shape[0], size=k, replace=False)].copy()
    labels = np.zeros(points.shape[0], dtype=np.int64)
    for _ in range(iters):
        dist = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)
        for j in range(k):
            if np.any(labels == j):
                centers[j] = points[labels == j].mean(axis=0)
    return centers, labels


def build_chunks(levels: list[dict], thresholds: list[float], train_cameras: list, vis_threshold: float) -> dict:
    cam_pos = np.stack([c.eye for c in train_cameras])
    d1 = max(thresholds[0], 1e-3)
    n_clusters = int(np.clip(round((4.0 / d1) * np.max(np.linalg.norm(cam_pos - cam_pos.mean(axis=0), axis=1))), 2, 8))
    centers, labels = kmeans(cam_pos, n_clusters)
    chunks = []
    for j, center in enumerate(centers):
        members = np.where(labels == j)[0]
        if len(members) == 0:
            member_cams = train_cameras
            radius = 0.0
        else:
            member_cams = [train_cameras[int(i)] for i in members]
            radius = float(np.max(np.linalg.norm(cam_pos[members] - center[None, :], axis=1)))
        active = active_indices_for_center(levels, thresholds, center, radius_offset=radius)
        filtered = []
        for scene, inds in zip(levels, active):
            if inds.size == 0:
                filtered.append(inds)
                continue
            scores = importance_scores(scene, member_cams, indices=inds)
            keep = scores >= vis_threshold
            if keep.sum() < max(4, int(0.25 * keep.size)):
                keep = scores >= np.quantile(scores, 0.55)
            filtered.append(inds[keep])
        chunks.append({"center": center, "radius": radius, "indices": filtered})
    return {"centers": centers, "chunks": chunks}


def render_chunk(levels: list[dict], chunk: dict, cam, opacity_scale: float = 1.0) -> tuple[np.ndarray, dict]:
    active_scene = gather_active_scene(levels, chunk["indices"])
    img, stats = render_scene(active_scene, cam, opacity_scale=opacity_scale, return_stats=True)
    return img, {"loaded": stats["loaded"], "visible": stats["visible"], "tile_cost": stats["tile_cost"]}


def render_opacity_blend(levels: list[dict], chunks: dict, cam) -> tuple[np.ndarray, dict]:
    centers = chunks["centers"]
    dist = np.linalg.norm(centers - cam.eye[None, :], axis=1)
    nearest = np.argsort(dist)[:2]
    d0, d1 = float(dist[nearest[0]]), float(dist[nearest[1]])
    w0 = d1 / max(d0 + d1, 1e-6)
    w1 = 1.0 - w0
    merged_parts = []
    opacity_parts = []
    c0 = chunks["chunks"][int(nearest[0])]["indices"]
    c1 = chunks["chunks"][int(nearest[1])]["indices"]
    for scene, inds0, inds1 in zip(levels, c0, c1):
        weight_by_idx: dict[int, float] = {}
        for idx in inds0:
            weight_by_idx[int(idx)] = weight_by_idx.get(int(idx), 0.0) + w0
        for idx in inds1:
            weight_by_idx[int(idx)] = weight_by_idx.get(int(idx), 0.0) + w1
        if not weight_by_idx:
            continue
        inds = np.array(sorted(weight_by_idx.keys()), dtype=np.int64)
        merged_parts.append({k: v[inds] for k, v in scene.items()})
        opacity_parts.append(np.array([min(1.0, weight_by_idx[int(i)]) for i in inds], dtype=np.float32))
    if not merged_parts:
        empty = {k: v[:0] for k, v in levels[0].items()}
        img, stats = render_scene(empty, cam, return_stats=True)
        return img, {"loaded": 0, "visible": 0, "tile_cost": 0.0}
    merged = {k: np.concatenate([p[k] for p in merged_parts], axis=0) for k in levels[0].keys()}
    opacity = np.concatenate(opacity_parts, axis=0)
    img, stats = render_scene(merged, cam, opacity_scale=opacity, return_stats=True)
    return img, {"loaded": stats["loaded"], "visible": stats["visible"], "tile_cost": stats["tile_cost"]}


def save_run(out_dir: Path, images: np.ndarray, stats: list[dict], config: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "renders.npz", images=images)
    scalar = [{k: v for k, v in st.items() if k != "importance"} for st in stats]
    (out_dir / "stats.json").write_text(json.dumps({"config": config, "per_view": scalar}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/lodge_synthetic"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/lodge"))
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.060)
    parser.add_argument("--visibility-threshold", type=float, default=0.030)
    parser.add_argument("--smooth-s", type=float, default=2.8)
    parser.add_argument("--thresholds", type=float, nargs="*", default=None)
    args = parser.parse_args()

    ensure_dataset(args.data_dir)
    scene, train, test, _ = load_dataset(args.data_dir)
    candidates = [3.0, 5.0, 7.0, 9.0]
    thresholds = args.thresholds or select_thresholds(scene, train, args.levels, candidates, args.gamma, args.smooth_s)
    levels = build_lod_levels(scene, train, thresholds, args.gamma, args.smooth_s)
    chunks = build_chunks(levels, thresholds, train, args.visibility_threshold)

    outputs = []
    stats = []
    for cam in test:
        img, st = render_opacity_blend(levels, chunks, cam)
        outputs.append(img)
        stats.append(st)
    save_run(
        args.out_dir,
        stack_renders(outputs),
        stats,
        {
            "method": "lodge",
            "thresholds": thresholds,
            "levels": args.levels,
            "gamma": args.gamma,
            "visibility_threshold": args.visibility_threshold,
            "smooth_s": args.smooth_s,
            "num_chunks": len(chunks["chunks"]),
            "level_gaussians": [int(x["means"].shape[0]) for x in levels],
        },
    )


if __name__ == "__main__":
    main()
