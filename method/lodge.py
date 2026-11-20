"""LODGE: level-of-detail Gaussian splatting with chunk-based rendering.

Implements the paper's pipeline (Sec. 3 and supp. A.2) on top of the public
base reconstruction G0:

1. Depth thresholds d_l are selected greedily (d_1 first, then d_2 > d_1, ...)
   by minimising the average number of Gaussians per 16x16 tile when rendering
   a subset of training views with LOD rendering (Eq. 2).
2. Each LOD level G_l is built from G0: a depth-aware 3D smoothing filter
   (Eq. 3, Mip-Splatting 3D filter with variance s*(d_l/f)^2 and opacity scaled
   by sqrt(det(Sigma)/det(Sigma + eps I))) is added to the Gaussian function, then
   three rounds of importance pruning (thresholds 0.2*gamma, 0.6*gamma, gamma;
   importance = maximum alpha-blending weight over training pixels) each followed
   by fine-tuning (L1 + D-SSIM) with LOD rendering over levels <= l, d_l drawn
   from U(0.7 d_l, 1.3 d_l), and gradients only into G_l.
3. Training camera positions are clustered with K-means into
   N = 4/d_1 * max_i ||c_i - mean(c)|| chunks (Eq. 5).  Each chunk stores the
   active set of Eq. (2) evaluated at its centre with depths increased by the
   chunk radius (distance to the closest other centre), followed by visibility
   filtering with importance computed from the chunk's training positions with
   the original and randomly sampled orientations.
4. The pool and chunk sets are exported in the benchmark representation format;
   the renderer blends the two closest chunks with the opacity interpolation of
   Eq. (4).  Ablation variants (nearest chunk only, per-frame dynamic LOD) are
   exported from the same build.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.gsrender import (
    Camera,
    default_device,
    load_base_gaussians,
    load_camera_sets,
    load_images,
    render_gaussians,
    save_representation,
)

KEYS = ("means", "scales", "quats", "opacities", "colors")


def log(msg: str) -> None:
    print(f"[lodge {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Levels and the 3D smoothing filter
# --------------------------------------------------------------------------


class Level:
    """Raw Gaussian parameters plus the level's 3D filter variance ``eps``."""

    def __init__(self, raw: dict, eps: float):
        self.raw = raw
        self.eps = float(eps)

    @property
    def n(self) -> int:
        return int(self.raw["means"].shape[0])

    def subset(self, keep: torch.Tensor) -> "Level":
        return Level({k: v[keep].detach().clone() for k, v in self.raw.items()}, self.eps)

    def effective(self) -> dict:
        return effective_params(self.raw["means"], self.raw["scales"], self.raw["quats"], self.raw["opacities"], self.raw["colors"], self.eps)


def effective_params(means, scales, quats, opacities, colors, eps: float) -> dict:
    """Gaussian function after convolving with an isotropic 3D filter of variance eps (Eq. 3)."""
    if eps <= 0:
        return {"means": means, "scales": scales, "quats": quats, "opacities": opacities, "colors": colors}
    s_eff = torch.sqrt(scales * scales + eps)
    comp = torch.prod(scales / s_eff, dim=1)
    return {"means": means, "scales": s_eff, "quats": quats, "opacities": opacities * comp, "colors": colors}


def filter_variance(depth: float, focal: float, s: float) -> float:
    return float(s * (depth / focal) ** 2)


# --------------------------------------------------------------------------
# LOD rendering (Eq. 2)
# --------------------------------------------------------------------------


def lod_render(levels_eff: list[dict], bounds: list[float], cam: Camera, return_importance: bool = False) -> dict:
    """Render the LOD stack: level j is active for camera distances in [bounds[j], bounds[j+1])."""
    eye = torch.as_tensor(cam.eye, dtype=torch.float32, device=levels_eff[0]["means"].device)
    parts = {k: [] for k in KEYS}
    sel = []
    for j, lvl in enumerate(levels_eff):
        d = (lvl["means"].detach() - eye).norm(dim=1)
        idx = torch.nonzero((d >= bounds[j]) & (d < bounds[j + 1]), as_tuple=False).squeeze(1)
        sel.append(idx)
        for k in KEYS:
            parts[k].append(lvl[k][idx])
    cat = {k: torch.cat(v, 0) for k, v in parts.items()}
    out = render_gaussians(cat["means"], cat["scales"], cat["quats"], cat["opacities"], cat["colors"], cam, return_importance=return_importance)
    if return_importance:
        per_level = []
        off = 0
        for j, idx in enumerate(sel):
            imp = torch.zeros(levels_eff[j]["means"].shape[0], device=eye.device)
            imp[idx] = out["importance"][off:off + idx.numel()]
            off += idx.numel()
            per_level.append(imp)
        out["level_importance"] = per_level
    return out


def bounds_for(thresholds: list[float]) -> list[float]:
    return [0.0] + list(thresholds) + [math.inf]


@torch.no_grad()
def level_importance(frozen_eff: list[dict], current: Level, thresholds: list[float], cams: list[Camera]) -> torch.Tensor:
    cur = current.effective()
    imp = torch.zeros(current.n, device=current.raw["means"].device)
    for cam in cams:
        out = lod_render(frozen_eff + [cur], bounds_for(thresholds), cam, return_importance=True)
        imp = torch.maximum(imp, out["level_importance"][-1])
    return imp


# --------------------------------------------------------------------------
# Fine-tuning (L1 + D-SSIM, only the current level is updated)
# --------------------------------------------------------------------------


def _ssim_window(device) -> torch.Tensor:
    x = torch.arange(11, dtype=torch.float32, device=device) - 5.0
    g = torch.exp(-(x * x) / (2 * 1.5 * 1.5))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    x = pred.permute(2, 0, 1)[None]
    y = target.permute(2, 0, 1)[None]
    mu_x = F.conv2d(x, window, padding=5, groups=3)
    mu_y = F.conv2d(y, window, padding=5, groups=3)
    sxx = F.conv2d(x * x, window, padding=5, groups=3) - mu_x * mu_x
    syy = F.conv2d(y * y, window, padding=5, groups=3) - mu_y * mu_y
    sxy = F.conv2d(x * y, window, padding=5, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sxx + syy + c2))
    return 1.0 - s.mean()


def pseudo_camera(rng: np.random.Generator, train: list[Camera]) -> Camera:
    base = train[int(rng.integers(len(train)))]
    eye = np.array(base.eye, dtype=np.float64) + np.array([rng.uniform(-1.0, 1.0), rng.uniform(-2.5, 2.5), rng.uniform(-0.5, 2.8)])
    eye[2] = max(eye[2], 1.0)
    return Camera(eye, float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(-0.17, 0.26)), base.f, base.width, base.height)


def finetune_level(
    current: Level,
    frozen_eff: list[dict],
    thresholds: list[float],
    cams: list[Camera],
    images: torch.Tensor,
    steps: int,
    seed: int,
    lrs: dict,
    pseudo_frac: float = 0.0,
    base_eff: dict | None = None,
) -> Level:
    if steps <= 0 or current.n == 0:
        return current
    device = current.raw["means"].device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    means = torch.nn.Parameter(current.raw["means"].clone())
    log_s = torch.nn.Parameter(torch.log(current.raw["scales"].clamp(min=1e-7)))
    quats = torch.nn.Parameter(current.raw["quats"].clone())
    op_logit = torch.nn.Parameter(torch.logit(current.raw["opacities"].clamp(1e-4, 1 - 1e-4)))
    col_logit = torch.nn.Parameter(torch.logit(current.raw["colors"].clamp(1e-4, 1 - 1e-4)))
    opt = torch.optim.Adam(
        [
            {"params": [means], "lr": lrs["means"]},
            {"params": [log_s], "lr": lrs["scales"]},
            {"params": [quats], "lr": lrs["quats"]},
            {"params": [op_logit], "lr": lrs["opacities"]},
            {"params": [col_logit], "lr": lrs["colors"]},
        ],
        eps=1e-15,
    )
    window = _ssim_window(device)
    for _ in range(steps):
        thr = list(thresholds)
        thr[-1] = float(thr[-1] * rng.uniform(0.7, 1.3))
        if pseudo_frac > 0 and rng.random() < pseudo_frac:
            cam = pseudo_camera(rng, cams)
            with torch.no_grad():
                target = render_gaussians(base_eff["means"], base_eff["scales"], base_eff["quats"], base_eff["opacities"], base_eff["colors"], cam)["image"]
        else:
            i = int(rng.integers(len(cams)))
            cam = cams[i]
            target = images[i]
        cur = effective_params(means, torch.exp(log_s), quats, torch.sigmoid(op_logit), torch.sigmoid(col_logit), current.eps)
        pred = lod_render(frozen_eff + [cur], bounds_for(thr), cam)["image"]
        if not pred.requires_grad:
            continue
        loss = 0.8 * torch.abs(pred - target).mean() + 0.2 * ssim_loss(pred, target, window)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    raw = {
        "means": means.detach(),
        "scales": torch.exp(log_s).detach(),
        "quats": quats.detach(),
        "opacities": torch.sigmoid(op_logit).detach(),
        "colors": torch.sigmoid(col_logit).detach(),
    }
    return Level(raw, current.eps)


# --------------------------------------------------------------------------
# Building LOD levels
# --------------------------------------------------------------------------


def build_levels(
    g0: Level,
    thresholds: list[float],
    cams: list[Camera],
    images: torch.Tensor | None,
    cfg: dict,
    steps: int,
    seed: int,
    prefix: list[Level] | None = None,
) -> list[Level]:
    """Build levels for ``thresholds`` from G0 (levels already in ``prefix`` are reused)."""
    levels = list(prefix) if prefix else [g0]
    focal = float(cams[0].f)
    base_eff = {k: v.detach() for k, v in g0.effective().items()}
    for l in range(len(levels), len(thresholds) + 1):
        d_l = thresholds[l - 1]
        lvl = Level({k: v.detach().clone() for k, v in g0.raw.items()}, filter_variance(d_l, focal, cfg["smooth_s"]))
        frozen = [{k: v.detach() for k, v in L.effective().items()} for L in levels]
        for r, mult in enumerate((0.2, 0.6, 1.0)):
            imp = level_importance(frozen, lvl, thresholds[:l], cams)
            lvl = lvl.subset(imp >= mult * cfg["gamma"])
            lvl = finetune_level(
                lvl, frozen, thresholds[:l], cams, images, steps, seed * 1000 + 10 * l + r, cfg["lrs"],
                pseudo_frac=cfg.get("pseudo_frac", 0.0), base_eff=base_eff,
            )
            log(f"level {l} (d={d_l:.1f}) round {r}: kept {lvl.n}")
        levels.append(lvl)
    return levels


@torch.no_grad()
def tile_cost(levels: list[Level], thresholds: list[float], cams: list[Camera]) -> float:
    eff = [L.effective() for L in levels]
    ntiles = (cams[0].width // 16) * (cams[0].height // 16)
    vals = [lod_render(eff, bounds_for(thresholds), cam)["tile_pairs"] / ntiles for cam in cams]
    return float(np.mean(vals))


def select_thresholds(g0: Level, cams: list[Camera], cost_cams: list[Camera], cfg: dict, n_levels: int) -> tuple[list[float], list]:
    """Greedy threshold search (Sec. 3, 'Selecting depth thresholds')."""
    selected: list[float] = []
    prefix = [g0]
    trace = []
    for _ in range(n_levels):
        best = None
        for cand in cfg["candidates"]:
            if selected and cand <= selected[-1]:
                continue
            thr = selected + [cand]
            levels = build_levels(g0, thr, cams, None, cfg, steps=0, seed=0, prefix=prefix)
            cost = tile_cost(levels, thr, cost_cams)
            trace.append({"thresholds": thr, "cost": cost})
            log(f"threshold candidate {thr}: avg Gaussians/tile {cost:.2f}")
            if best is None or cost < best[0]:
                best = (cost, cand, levels)
        if best is None:
            break
        selected.append(float(best[1]))
        prefix = best[2]
    return selected, trace


# --------------------------------------------------------------------------
# Chunks (K-means on camera positions, Eq. 5) and visibility filtering
# --------------------------------------------------------------------------


def kmeans(points: np.ndarray, k: int, seed: int = 0, iters: int = 100, n_init: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """K-means (k-means++ init) with ``n_init`` restarts; returns the lowest-inertia solution."""
    best = None
    for r in range(n_init):
        rng = np.random.default_rng(seed + r)
        centers = [points[int(rng.integers(len(points)))]]
        for _ in range(1, k):
            d2 = np.min(((points[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(len(points), 1.0 / len(points))
            centers.append(points[int(rng.choice(len(points), p=p))])
        centers = np.array(centers, dtype=np.float64)
        labels = np.zeros(len(points), dtype=np.int64)
        for _ in range(iters):
            labels = np.argmin(((points[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
            new = centers.copy()
            for j in range(k):
                if np.any(labels == j):
                    new[j] = points[labels == j].mean(axis=0)
            if np.allclose(new, centers):
                break
            centers = new
        inertia = float(((points - centers[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, centers, labels)
    order = np.lexsort((best[1][:, 1], best[1][:, 0]))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return best[1][order], remap[best[2]]


def random_orientations(cams: list[Camera], per_camera: int, min_total: int, rng: np.random.Generator) -> list[Camera]:
    """Original training views plus views at the same positions with random orientations.

    At least ``min_total`` random-orientation views are drawn per chunk so the
    whole sphere of directions around the chunk is covered.
    """
    out = list(cams)
    n_rand = max(per_camera * len(cams), min_total)
    for r in range(n_rand):
        cam = cams[r % len(cams)]
        out.append(Camera(cam.eye, float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(-0.35, 0.6)), cam.f, cam.width, cam.height))
    return out


@torch.no_grad()
def build_chunks(levels: list[Level], thresholds: list[float], train: list[Camera], cfg: dict) -> dict:
    """Chunks are a deterministic function of the training camera positions and the levels."""
    pos = np.stack([c.eye for c in train])
    spread = float(np.max(np.linalg.norm(pos - pos.mean(0), axis=1)))
    n_clusters = max(2, int(round(4.0 / thresholds[0] * spread)))
    n_clusters = min(n_clusters, len(pos))
    centers, labels = kmeans(pos, n_clusters, seed=0)
    cd = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    np.fill_diagonal(cd, np.inf)
    eff = [L.effective() for L in levels]
    chunks = []
    for j, c in enumerate(centers):
        rng = np.random.default_rng(1000 + j)
        radius = float(cd[j].min())
        # Depths are offset by the chunk radius so that every camera position in
        # the chunk gets at least the resolution Eq. (2) would give it (finer
        # levels extend further; paper Sec. 3 and Table 3 "finer LOD resolutions").
        d = [0.0] + [t + radius for t in thresholds] + [math.inf]
        ct = torch.as_tensor(c, dtype=torch.float32, device=eff[0]["means"].device)
        active = []
        for l, e in enumerate(eff):
            dist = (e["means"] - ct).norm(dim=1)
            active.append(torch.nonzero((dist >= d[l]) & (dist < d[l + 1]), as_tuple=False).squeeze(1))
        members = [train[i] for i in np.where(labels == j)[0]]
        if not members:
            nearest = np.argsort(np.linalg.norm(pos - c[None], axis=1))[:3]
            members = [train[int(i)] for i in nearest]
        views = random_orientations(members, cfg["vis_random_views"], cfg["vis_min_views"], rng)
        cat = {k: torch.cat([e[k][a] for e, a in zip(eff, active)], 0) for k in KEYS}
        imp = torch.zeros(cat["means"].shape[0], device=ct.device)
        for cam in views:
            out = render_gaussians(cat["means"], cat["scales"], cat["quats"], cat["opacities"], cat["colors"], cam, return_importance=True)
            imp = torch.maximum(imp, out["importance"])
        keep = imp >= cfg["vis_threshold"]
        kept, off = [], 0
        for a in active:
            kept.append(a[keep[off:off + a.numel()]].cpu().numpy())
            off += a.numel()
        chunks.append({"center": c, "radius": radius, "indices": kept, "members": len(members)})
        log(f"chunk {j}: radius {radius:.2f} m, members {len(members)}, active {sum(x.numel() for x in active)} -> {sum(k.size for k in kept)}")
    return {"centers": centers, "chunks": chunks, "n_clusters": n_clusters}


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def export_chunked(levels: list[Level], chunks: dict, path: Path, blend: bool) -> dict:
    eff = [{k: v.detach().cpu().numpy() for k, v in L.effective().items()} for L in levels]
    offsets = np.cumsum([0] + [e["means"].shape[0] for e in eff])
    used = np.zeros(offsets[-1], dtype=bool)
    for ch in chunks["chunks"]:
        for l, idx in enumerate(ch["indices"]):
            used[offsets[l] + idx] = True
    remap = -np.ones(offsets[-1], dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    pool = {k: np.concatenate([e[k] for e in eff])[used] for k in KEYS}
    lists = []
    for ch in chunks["chunks"]:
        lists.append(np.concatenate([remap[offsets[l] + idx] for l, idx in enumerate(ch["indices"])]))
    save_representation(path, pool, chunks["centers"], lists, blend_two_nearest=blend)
    return {"pool": int(used.sum()), "chunk_sizes": [int(x.size) for x in lists]}


def export_dynamic(levels: list[Level], thresholds: list[float], center: np.ndarray, path: Path) -> dict:
    eff = [{k: v.detach().cpu().numpy() for k, v in L.effective().items()} for L in levels]
    pool = {k: np.concatenate([e[k] for e in eff]) for k in KEYS}
    b = bounds_for(thresholds)
    lo = np.concatenate([np.full(e["means"].shape[0], b[l], dtype=np.float32) for l, e in enumerate(eff)])
    hi = np.concatenate([np.full(e["means"].shape[0], b[l + 1], dtype=np.float32) for l, e in enumerate(eff)])
    n = pool["means"].shape[0]
    save_representation(path, pool, center[None], [np.arange(n)], blend_two_nearest=False, band_lo=lo, band_hi=hi)
    return {"pool": int(n)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


DEFAULTS = {
    "levels": 2,
    "gamma": 0.03,
    "smooth_s": 0.2,
    "vis_threshold": 0.02,
    "vis_random_views": 2,
    "vis_min_views": 32,
    "finetune_steps": 1000,
    "candidates": [4.0, 6.0, 8.0, 11.0, 15.0, 20.0, 27.0, 36.0, 48.0, 64.0],
    "selection_views": 32,
    "lrs": {"means": 2e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 2.5e-2, "colors": 5e-3},
    "pseudo_frac": 0.0,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/lodge_city"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/lodge"))
    parser.add_argument("--ablation-dir", type=Path, default=None, help="also export nearest-chunk and dynamic-LOD variants here")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", type=int, default=DEFAULTS["levels"])
    parser.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    parser.add_argument("--smooth-s", type=float, default=DEFAULTS["smooth_s"])
    parser.add_argument("--vis-threshold", type=float, default=DEFAULTS["vis_threshold"])
    parser.add_argument("--finetune-steps", type=int, default=DEFAULTS["finetune_steps"])
    parser.add_argument("--pseudo-frac", type=float, default=DEFAULTS["pseudo_frac"])
    parser.add_argument("--thresholds", type=float, nargs="*", default=None, help="manual thresholds (skip automatic selection)")
    parser.add_argument("--threshold-cache", type=Path, default=None, help="JSON cache for the deterministic threshold search")
    args = parser.parse_args()

    device = default_device()
    cfg = dict(DEFAULTS)
    cfg.update({"levels": args.levels, "gamma": args.gamma, "smooth_s": args.smooth_s, "vis_threshold": args.vis_threshold,
                "finetune_steps": args.finetune_steps, "pseudo_frac": args.pseudo_frac})
    t0 = time.time()
    sets = load_camera_sets(args.data_dir)
    train = sets["train"]
    images = torch.as_tensor(load_images(args.data_dir, "train"), device=device).float() / 255.0
    base = load_base_gaussians(args.data_dir)
    g0 = Level({k: torch.as_tensor(v, device=device) for k, v in base.items()}, 0.0)
    log(f"device={device} base Gaussians={g0.n} train views={len(train)} cfg={json.dumps({k: cfg[k] for k in ('levels','gamma','smooth_s','vis_threshold','finetune_steps','pseudo_frac')})}")

    trace = []
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds]
    else:
        key = json.dumps({"levels": cfg["levels"], "gamma": cfg["gamma"], "s": cfg["smooth_s"], "cands": cfg["candidates"], "views": cfg["selection_views"], "n0": g0.n})
        cache = json.loads(args.threshold_cache.read_text()) if args.threshold_cache and args.threshold_cache.exists() else {}
        if key in cache:
            thresholds, trace = cache[key]["thresholds"], cache[key]["trace"]
            log(f"thresholds from cache: {thresholds}")
        else:
            step = max(1, len(train) // cfg["selection_views"])
            cost_cams = train[::step]
            thresholds, trace = select_thresholds(g0, train, cost_cams, cfg, cfg["levels"])
            if args.threshold_cache:
                cache[key] = {"thresholds": thresholds, "trace": trace}
                args.threshold_cache.parent.mkdir(parents=True, exist_ok=True)
                args.threshold_cache.write_text(json.dumps(cache, indent=1))
    log(f"thresholds: {thresholds}")

    levels = build_levels(g0, thresholds, train, images, cfg, steps=cfg["finetune_steps"], seed=args.seed)
    chunks = build_chunks(levels, thresholds, train, cfg)
    info = export_chunked(levels, chunks, args.out_dir / "representation.npz", blend=True)
    meta = {
        "method": "lodge",
        "seed": args.seed,
        "config": {k: cfg[k] for k in ("levels", "gamma", "smooth_s", "vis_threshold", "vis_random_views", "vis_min_views", "finetune_steps", "pseudo_frac", "lrs")},
        "thresholds": thresholds,
        "threshold_search": trace,
        "level_sizes": [L.n for L in levels],
        "n_chunks": chunks["n_clusters"],
        "pool": info["pool"],
        "mean_chunk_size": float(np.mean(info["chunk_sizes"])),
        "seconds": time.time() - t0,
        "note": "Informational only; the evaluator recomputes every scored number from representation.npz.",
    }
    (args.out_dir / "method_info.json").write_text(json.dumps(meta, indent=1))
    log(f"wrote {args.out_dir} pool={info['pool']} chunks={chunks['n_clusters']} in {meta['seconds']:.0f}s")
    if args.ablation_dir:
        export_chunked(levels, chunks, args.ablation_dir / "nearest_chunk" / "representation.npz", blend=False)
        center = np.stack([c.eye for c in train]).mean(0)
        export_dynamic(levels, thresholds, center, args.ablation_dir / "lod_dynamic" / "representation.npz")
        log(f"wrote ablation variants to {args.ablation_dir}")


if __name__ == "__main__":
    main()
