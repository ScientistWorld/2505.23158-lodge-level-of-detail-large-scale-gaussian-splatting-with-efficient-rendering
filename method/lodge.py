"""Compact implementation of the LODGE Gaussian-splatting algorithm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.gaussian_scene import ensure_dataset, load_dataset, load_train_targets, render_scene, stack_renders


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


def _camera_basis_tensors(cam) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v if n < 1e-8 else v / n

    forward = norm(cam.target - cam.eye)
    right = norm(np.cross(forward, cam.up))
    up = norm(np.cross(right, forward))
    return (
        torch.as_tensor(right, dtype=torch.float32),
        torch.as_tensor(up, dtype=torch.float32),
        torch.as_tensor(forward, dtype=torch.float32),
    )


def differentiable_render(
    means: torch.Tensor,
    sigma: torch.Tensor,
    opacity: torch.Tensor,
    colors: torch.Tensor,
    cam,
) -> torch.Tensor:
    """Small autograd renderer for the LOD fine-tuning stage.

    The production renderer in data/ keeps exact stats and tile accounting.
    This renderer mirrors the same front-to-back Gaussian alpha compositing but
    evaluates full image footprints, which is compact and differentiable for
    the CPU-scale benchmark.
    """

    right, up, forward = _camera_basis_tensors(cam)
    eye = torch.as_tensor(cam.eye, dtype=torch.float32)
    rel = means - eye[None, :]
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    valid = z > 0.35
    order = torch.argsort(z.detach())

    yy, xx = torch.meshgrid(
        torch.arange(cam.height, dtype=torch.float32),
        torch.arange(cam.width, dtype=torch.float32),
        indexing="ij",
    )
    img = torch.zeros((cam.height, cam.width, 3), dtype=torch.float32)
    trans = torch.ones((cam.height, cam.width), dtype=torch.float32)
    bg = torch.tensor([0.02, 0.025, 0.03], dtype=torch.float32)

    for j_t in order:
        j = int(j_t)
        if not bool(valid[j]):
            continue
        zj = torch.clamp(z[j], min=1e-5)
        u = cam.f * x[j] / zj + cam.width * 0.5
        v = -cam.f * y[j] / zj + cam.height * 0.5
        sig = torch.clamp(cam.f * sigma[j] / zj, min=0.45)
        g = torch.exp(-0.5 * (((xx - u) ** 2 + (yy - v) ** 2) / (sig * sig)))
        alpha = torch.clamp(opacity[j] * g, 0.0, 0.98)
        contrib = trans * alpha
        img = img + contrib[..., None] * colors[j]
        trans = trans * (1.0 - alpha)
    return torch.clamp(img + trans[..., None] * bg, 0.0, 1.0)


def differentiable_render_lod_stack(
    frozen_levels: list[dict],
    current_tensors: dict[str, torch.Tensor],
    current_means_np: np.ndarray,
    thresholds: list[float],
    cam,
) -> torch.Tensor:
    """Render active LOD levels up to the current optimized level.

    The paper fine-tunes level l with standard LOD rendering over levels
    0..l, while gradients update only G_l.  Frozen finer levels enter the
    compositing graph as constants; the active set itself follows Eq. (2).
    """

    d = [0.0] + list(thresholds) + [np.inf]
    means_parts = []
    sigma_parts = []
    opacity_parts = []
    color_parts = []
    center = cam.eye

    for lvl, scene in enumerate(frozen_levels):
        dist = np.linalg.norm(scene["means"] - center[None, :], axis=1)
        inds = np.where((dist >= d[lvl]) & (dist < d[lvl + 1]))[0]
        if inds.size == 0:
            continue
        means_parts.append(torch.as_tensor(scene["means"][inds], dtype=torch.float32))
        sigma_parts.append(torch.as_tensor(scene["sigma"][inds], dtype=torch.float32))
        opacity_parts.append(torch.as_tensor(scene["opacity"][inds], dtype=torch.float32))
        color_parts.append(torch.as_tensor(scene["colors"][inds], dtype=torch.float32))

    current_lvl = len(frozen_levels)
    dist = np.linalg.norm(current_means_np - center[None, :], axis=1)
    inds_np = np.where((dist >= d[current_lvl]) & (dist < d[current_lvl + 1]))[0]
    if inds_np.size:
        inds = torch.as_tensor(inds_np, dtype=torch.long)
        means_parts.append(current_tensors["means"][inds])
        sigma_parts.append(current_tensors["sigma"][inds])
        opacity_parts.append(current_tensors["opacity"][inds])
        color_parts.append(current_tensors["colors"][inds])

    if not means_parts:
        empty = torch.empty((0, 3), dtype=torch.float32)
        return differentiable_render(empty, torch.empty(0), torch.empty(0), empty, cam)

    return differentiable_render(
        torch.cat(means_parts, dim=0),
        torch.cat(sigma_parts, dim=0),
        torch.cat(opacity_parts, dim=0),
        torch.cat(color_parts, dim=0),
        cam,
    )


def dssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    x = pred.reshape(-1, 3)
    y = target.reshape(-1, 3)
    vals = []
    for ch in range(3):
        a = x[:, ch]
        b = y[:, ch]
        mux = a.mean()
        muy = b.mean()
        vx = a.var(unbiased=False)
        vy = b.var(unbiased=False)
        cov = ((a - mux) * (b - muy)).mean()
        ssim = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2))
        vals.append(ssim)
    return (1.0 - torch.stack(vals).mean()) * 0.5


def gradient_finetune_lod(
    scene: dict,
    frozen_levels: list[dict],
    thresholds: list[float],
    cameras: list,
    train_gt: np.ndarray,
    steps: int,
    lr: float,
    seed: int,
) -> dict:
    """Fine-tune a pruned LOD level with the paper's DSSIM+L1 objective.

    The full paper uses 1000 CUDA optimization iterations after each pruning
    round.  This compact reproduction scales the iteration count down but keeps
    the same optimization stage and loss family on surviving LOD Gaussians.
    """

    if steps <= 0 or scene["means"].shape[0] == 0:
        return scene

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    means = torch.as_tensor(scene["means"], dtype=torch.float32)
    sigma_floor = torch.as_tensor(scene["sigma"], dtype=torch.float32)
    sigma_delta_raw = torch.nn.Parameter(torch.full_like(sigma_floor, -8.0))
    opacity_init = torch.clamp(torch.as_tensor(scene["opacity"], dtype=torch.float32) / 0.98, 1e-4, 1 - 1e-4)
    opacity_raw = torch.nn.Parameter(torch.logit(opacity_init))
    color_raw = torch.nn.Parameter(torch.logit(torch.clamp(torch.as_tensor(scene["colors"], dtype=torch.float32), 1e-4, 1 - 1e-4)))
    opt = torch.optim.Adam([sigma_delta_raw, opacity_raw, color_raw], lr=lr)

    targets = torch.as_tensor(train_gt, dtype=torch.float32)
    level_idx = len(frozen_levels)
    for step in range(steps):
        cam_idx = (step + seed) % len(cameras)
        active_thresholds = list(thresholds)
        if active_thresholds:
            active_thresholds[-1] = float(rng.uniform(0.7 * active_thresholds[-1], 1.3 * active_thresholds[-1]))
        current_tensors = {
            "means": means,
            "sigma": sigma_floor + torch.nn.functional.softplus(sigma_delta_raw),
            "opacity": torch.sigmoid(opacity_raw) * 0.98,
            "colors": torch.sigmoid(color_raw),
        }
        pred = differentiable_render_lod_stack(
            frozen_levels[:level_idx],
            current_tensors,
            scene["means"],
            active_thresholds,
            cameras[cam_idx],
        )
        if not pred.requires_grad:
            continue
        target = targets[cam_idx]
        loss = torch.mean(torch.abs(pred - target)) + dssim_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return {
        "means": scene["means"],
        "sigma": (sigma_floor + torch.nn.functional.softplus(sigma_delta_raw)).detach().numpy().astype(np.float32),
        "opacity": (torch.sigmoid(opacity_raw) * 0.98).detach().numpy().astype(np.float32),
        "colors": torch.sigmoid(color_raw).detach().numpy().astype(np.float32),
    }


def build_lod_levels(
    scene: dict,
    cameras: list,
    train_gt: np.ndarray,
    thresholds: list[float],
    gamma: float,
    smooth_s: float,
    finetune_steps: int,
    finetune_lr: float,
) -> list[dict]:
    levels = [copy_scene(scene)]
    focal = float(np.mean([c.f for c in cameras]))
    for level_idx, depth in enumerate(thresholds, start=1):
        lod = smooth_scene(scene, depth, focal, smooth_s)
        for round_idx, mult in enumerate([0.2, 0.6, 1.0]):
            lod = prune_by_importance(lod, cameras, gamma * mult)
            lod = gradient_finetune_lod(
                lod,
                levels,
                thresholds[:level_idx],
                cameras,
                train_gt,
                steps=finetune_steps,
                lr=finetune_lr,
                seed=level_idx * 100 + round_idx,
            )
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


def threshold_cost(scene: dict, cameras: list, train_gt: np.ndarray, thresholds: list[float], gamma: float, smooth_s: float) -> float:
    levels = build_lod_levels(
        scene,
        cameras[:3],
        train_gt[:3],
        thresholds,
        gamma=gamma,
        smooth_s=smooth_s,
        finetune_steps=0,
        finetune_lr=0.0,
    )
    costs = []
    for cam in cameras[:3]:
        _, stats = render_lod_at_camera(levels, thresholds, cam)
        costs.append(stats["tile_cost"])
    return float(np.mean(costs))


def select_thresholds(
    scene: dict,
    cameras: list,
    train_gt: np.ndarray,
    n_levels: int,
    candidates: list[float],
    gamma: float,
    smooth_s: float,
) -> list[float]:
    selected: list[float] = []
    for _ in range(n_levels):
        best = None
        for cand in candidates:
            if selected and cand <= selected[-1] + 0.5:
                continue
            trial = selected + [cand]
            cost = threshold_cost(scene, cameras, train_gt, trial, gamma=gamma, smooth_s=smooth_s)
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


def random_orientation_cameras(cameras: list, seed: int, per_camera: int = 2) -> list:
    rng = np.random.default_rng(seed)
    augmented = list(cameras)
    for cam in cameras:
        for _ in range(per_camera):
            direction = rng.normal(size=3).astype(np.float32)
            direction[2] *= 0.35
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                direction = direction / norm
            target = cam.eye + direction * 8.0
            augmented.append(
                cam.__class__(
                    np.array(cam.eye, copy=True),
                    target.astype(np.float32),
                    np.array(cam.up, copy=True),
                    cam.f,
                    cam.width,
                    cam.height,
                )
            )
    return augmented


def build_chunks(levels: list[dict], thresholds: list[float], train_cameras: list, vis_threshold: float) -> dict:
    cam_pos = np.stack([c.eye for c in train_cameras])
    d1 = max(thresholds[0], 1e-3)
    n_clusters = int(np.clip(round((4.0 / d1) * np.max(np.linalg.norm(cam_pos - cam_pos.mean(axis=0), axis=1))), 2, 8))
    centers, labels = kmeans(cam_pos, n_clusters)
    center_dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(center_dist, np.inf)
    chunks = []
    for j, center in enumerate(centers):
        members = np.where(labels == j)[0]
        if len(members) == 0:
            member_cams = train_cameras
        else:
            member_cams = [train_cameras[int(i)] for i in members]
        radius = float(np.min(center_dist[j])) if n_clusters > 1 else 0.0
        active = active_indices_for_center(levels, thresholds, center, radius_offset=radius)
        visibility_cams = random_orientation_cameras(member_cams, seed=1000 + j)
        filtered = []
        for scene, inds in zip(levels, active):
            if inds.size == 0:
                filtered.append(inds)
                continue
            scores = importance_scores(scene, visibility_cams, indices=inds)
            keep = scores >= vis_threshold
            if keep.sum() < max(4, int(0.10 * keep.size)):
                keep = scores >= np.quantile(scores, 0.80)
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
    merged_parts = []
    opacity_parts = []
    idx0 = int(nearest[0])
    idx1 = int(nearest[1])
    c0 = chunks["chunks"][idx0]["indices"]
    c1 = chunks["chunks"][idx1]["indices"]
    m0 = centers[idx0]
    m1 = centers[idx1]
    denom = float(np.sum((m0 - m1) ** 2))
    if denom <= 1e-8:
        t0 = t1 = 1.0
    else:
        # Eq. (4): project the camera onto the line between chunk centers.
        t0 = float(np.clip(np.dot(cam.eye - m1, m0 - m1) / denom, 0.0, 1.0))
        t1 = float(np.clip(np.dot(cam.eye - m0, m1 - m0) / denom, 0.0, 1.0))
    for scene, inds0, inds1 in zip(levels, c0, c1):
        set0 = {int(i) for i in inds0}
        set1 = {int(i) for i in inds1}
        union = sorted(set0 | set1)
        if not union:
            continue
        weights = []
        for idx in union:
            if idx in set0 and idx in set1:
                weights.append(1.0)
            elif idx in set0:
                weights.append(t0)
            else:
                weights.append(t1)
        inds = np.array(union, dtype=np.int64)
        merged_parts.append({k: v[inds] for k, v in scene.items()})
        opacity_parts.append(np.array(weights, dtype=np.float32))
    if not merged_parts:
        empty = {k: v[:0] for k, v in levels[0].items()}
        img, stats = render_scene(empty, cam, return_stats=True)
        return img, {"loaded": 0, "visible": 0, "tile_cost": 0.0}
    merged = {k: np.concatenate([p[k] for p in merged_parts], axis=0) for k in levels[0].keys()}
    opacity = np.concatenate(opacity_parts, axis=0)
    img, stats = render_scene(merged, cam, opacity_scale=opacity, return_stats=True)
    return img, {"loaded": stats["loaded"], "visible": stats["visible"], "tile_cost": stats["tile_cost"]}


def render_nearest_chunk(levels: list[dict], chunks: dict, cam) -> tuple[np.ndarray, dict]:
    centers = chunks["centers"]
    dist = np.linalg.norm(centers - cam.eye[None, :], axis=1)
    nearest = int(np.argmin(dist))
    return render_chunk(levels, chunks["chunks"][nearest], cam)


def render_variant(levels: list[dict], thresholds: list[float], chunks: dict, cam, variant: str) -> tuple[np.ndarray, dict]:
    if variant == "lod_dynamic":
        return render_lod_at_camera(levels, thresholds, cam)
    if variant == "nearest_chunk":
        return render_nearest_chunk(levels, chunks, cam)
    if variant == "opacity_blend":
        return render_opacity_blend(levels, chunks, cam)
    raise ValueError(f"Unknown LODGE variant: {variant}")


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
    parser.add_argument("--gamma", type=float, default=0.055)
    parser.add_argument("--visibility-threshold", type=float, default=0.220)
    parser.add_argument("--smooth-s", type=float, default=1.0)
    parser.add_argument("--finetune-steps", type=int, default=12)
    parser.add_argument("--finetune-lr", type=float, default=0.035)
    parser.add_argument("--thresholds", type=float, nargs="*", default=None)
    parser.add_argument(
        "--variant",
        choices=["opacity_blend", "nearest_chunk", "lod_dynamic"],
        default="opacity_blend",
        help="Rendering mode: full LODGE, nearest precomputed chunk without blending, or per-camera dynamic LOD.",
    )
    args = parser.parse_args()

    ensure_dataset(args.data_dir)
    scene, train, test, _ = load_dataset(args.data_dir)
    train_gt = load_train_targets(args.data_dir)
    candidates = [3.0, 5.0, 7.0, 9.0]
    thresholds = args.thresholds or select_thresholds(scene, train, train_gt, args.levels, candidates, args.gamma, args.smooth_s)
    levels = build_lod_levels(
        scene,
        train,
        train_gt,
        thresholds,
        args.gamma,
        args.smooth_s,
        args.finetune_steps,
        args.finetune_lr,
    )
    chunks = build_chunks(levels, thresholds, train, args.visibility_threshold)

    outputs = []
    stats = []
    for cam in test:
        img, st = render_variant(levels, thresholds, chunks, cam, args.variant)
        outputs.append(img)
        stats.append(st)
    save_run(
        args.out_dir,
        stack_renders(outputs),
        stats,
        {
            "method": args.variant,
            "thresholds": thresholds,
            "levels": args.levels,
            "gamma": args.gamma,
            "visibility_threshold": args.visibility_threshold,
            "smooth_s": args.smooth_s,
            "finetune_steps": args.finetune_steps,
            "finetune_lr": args.finetune_lr,
            "num_chunks": len(chunks["chunks"]),
            "level_gaussians": [int(x["means"].shape[0]) for x in levels],
        },
    )


if __name__ == "__main__":
    main()
