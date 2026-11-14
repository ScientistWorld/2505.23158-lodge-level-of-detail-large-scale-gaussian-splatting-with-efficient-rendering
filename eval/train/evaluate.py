"""Visible (train-slice) evaluator.

Mirrors the hidden test protocol on the public validation cameras in
``data/lodge_city``: every method directory under ``checkpoints/`` (plus the
required ``lodge`` entry) is scored by rendering its ``representation.npz``
with the public renderer ``data/gsrender.py``.  Loaded/visible counts and
images are recomputed here; nothing else the method writes is read.

Gates (identical to the hidden test): an invalid or missing representation, a
peak resident set larger than the full representation, or a PSNR more than the
tolerance below the full representation in any experiment marks the method as
failed, and it receives the worst value of every metric in every experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from data import gsrender as R

BASELINE = "full_representation"
REQUIRED = ("lodge",)
POOL_FACTOR = 4
EXPERIMENTS = {
    "trajectory_views": {"set": "val_on_path", "kind": "views"},
    "flythrough": {"set": "val_flythrough", "kind": "path"},
    "off_path_views": {"set": "val_off_path", "kind": "views"},
}
QUALITY_TOLERANCE_DB = {"trajectory_views": 1.0, "flythrough": 1.0, "off_path_views": 1.5}
VIEW_METRICS = ("psnr", "loaded_gaussians", "visible_gaussians", "ssim", "worst_view_psnr", "peak_loaded_gaussians", "tile_pairs")
PATH_METRICS = ("transition_psnr", "psnr", "ssim", "loaded_gaussians", "visible_gaussians", "peak_loaded_gaussians")
HIGHER_IS_BETTER = {"psnr", "ssim", "worst_view_psnr", "transition_psnr"}


def log(msg: str) -> None:
    print(f"[eval-train {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_public(data_dir: Path) -> dict:
    cams = R.load_camera_sets(data_dir)
    return {exp: (cams[spec["set"]], R.load_images(data_dir, spec["set"])) for exp, spec in EXPERIMENTS.items()}


def measure(rep: dict, rep_t: dict, cams: list, gts: np.ndarray, kind: str) -> dict:
    preds, loaded, visible, pairs = [], [], [], []
    for cam in cams:
        out = R.render_representation(rep, rep_t, cam)
        preds.append(R.to_uint8(out["image"]))
        loaded.append(out["loaded"])
        visible.append(out["visible"])
        pairs.append(out["tile_pairs"])
    psnrs = [R.psnr_u8(p, g) for p, g in zip(preds, gts)]
    m = {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean([R.ssim_u8(p, g) for p, g in zip(preds, gts)])),
        "worst_view_psnr": float(np.min(psnrs)),
        "loaded_gaussians": float(np.mean(loaded)),
        "visible_gaussians": float(np.mean(visible)),
        "peak_loaded_gaussians": float(np.max(loaded)),
        "tile_pairs": float(np.mean(pairs)),
    }
    if kind == "path":
        m["transition_psnr"] = R.transition_psnr(preds, list(gts))
    return m


def metric_names(kind: str) -> tuple:
    return PATH_METRICS if kind == "path" else VIEW_METRICS


def worst_values(kind: str, worst_count: float) -> dict:
    return {k: (0.0 if k in HIGHER_IS_BETTER else worst_count * (64.0 if k == "tile_pairs" else 1.0)) for k in metric_names(kind)}


def discover(checkpoints: Path) -> list[str]:
    names = set(REQUIRED)
    if checkpoints.is_dir():
        for d in sorted(checkpoints.iterdir()):
            if d.is_dir() and not d.name.startswith(("_", ".")) and d.name != BASELINE:
                names.add(d.name)
    return sorted(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/lodge_city"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = R.default_device()
    sets = load_public(args.data_dir)
    base = R.load_base_gaussians(args.data_dir)
    n0 = int(base["means"].shape[0])
    max_pool = POOL_FACTOR * n0
    log(f"device={device} full representation={n0} Gaussians")

    base_rep = dict(base)
    base_rep.update({
        "band_lo": np.zeros(n0, dtype=np.float32), "band_hi": np.full(n0, np.inf, dtype=np.float32),
        "chunk_centers": np.zeros((1, 3)), "chunks": [np.arange(n0)], "blend_two_nearest": 0,
    })
    base_t = R.representation_to_torch(base_rep, device)
    baseline = {exp: measure(base_rep, base_t, *sets[exp], EXPERIMENTS[exp]["kind"]) for exp in EXPERIMENTS}
    results = {exp: {BASELINE: {k: baseline[exp][k] for k in metric_names(EXPERIMENTS[exp]["kind"])}} for exp in EXPERIMENTS}
    log("baseline: " + json.dumps({e: round(v["psnr"], 3) for e, v in baseline.items()}))

    status = {}
    for name in discover(args.checkpoints_dir):
        path = args.checkpoints_dir / name / "representation.npz"
        reason, measured = None, None
        try:
            rep = R.load_representation(path, max_gaussians=max_pool)
            rep_t = R.representation_to_torch(rep, device)
            measured = {exp: measure(rep, rep_t, *sets[exp], EXPERIMENTS[exp]["kind"]) for exp in EXPERIMENTS}
            for exp in EXPERIMENTS:
                if not all(math.isfinite(v) for v in measured[exp].values()):
                    reason = f"non-finite metric in {exp}"
            peak = max(measured[e]["peak_loaded_gaussians"] for e in EXPERIMENTS)
            if reason is None and peak > n0:
                reason = f"memory: peak resident Gaussians {peak:.0f} exceed the full representation ({n0})"
            if reason is None:
                for exp in EXPERIMENTS:
                    floor = baseline[exp]["psnr"] - QUALITY_TOLERANCE_DB[exp]
                    if measured[exp]["psnr"] < floor:
                        reason = f"quality: {exp} PSNR {measured[exp]['psnr']:.3f} below floor {floor:.3f}"
                        break
        except R.RepresentationError as exc:
            reason = f"invalid representation: {exc}"
        except Exception as exc:  # noqa: BLE001
            reason = f"evaluation error: {exc.__class__.__name__}: {exc}"
            traceback.print_exc()
        status[name] = {"passed": reason is None, "reason": reason or "ok"}
        if measured is not None:
            status[name]["measured"] = measured
        for exp in EXPERIMENTS:
            kind = EXPERIMENTS[exp]["kind"]
            results[exp][name] = {k: measured[exp][k] for k in metric_names(kind)} if reason is None else worst_values(kind, float(max_pool))
        log(f"{name}: {status[name]['reason']}" + ("" if measured is None else " " + json.dumps({e: round(measured[e]['psnr'], 3) for e in EXPERIMENTS})))

    scores = {
        "_meta": {"slice": "train", "full_representation_gaussians": n0, "quality_tolerance_db": QUALITY_TOLERANCE_DB, "status": status},
        "experiments": {
            exp: {"_meta": {"cameras": len(sets[exp][0]), "kind": EXPERIMENTS[exp]["kind"]}, "results": results[exp]} for exp in EXPERIMENTS
        },
    }
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.scores.with_suffix(".tmp")
    tmp.write_text(json.dumps(scores, indent=2))
    os.replace(tmp, args.scores)
    log(f"wrote {args.scores}")


if __name__ == "__main__":
    main()
