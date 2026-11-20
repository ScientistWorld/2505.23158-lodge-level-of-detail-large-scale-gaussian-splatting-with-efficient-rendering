"""Hidden test evaluator.

Scores every method directory under ``checkpoints/`` (plus the required
``lodge`` entry, which scores as failed when absent) by rendering its
``representation.npz`` at hidden cameras with this directory's own copy of the
renderer.  Every number is recomputed from the representation: images, loaded
(resident) Gaussians, visible Gaussians, tile work.  Nothing the method writes
besides ``representation.npz`` is read.

The full-representation baseline is this directory's own copy of the trained
base reconstruction (stored in the hidden set), so it cannot be altered by
editing public files.

A method fails (and receives the worst value of every metric in every
experiment) when its representation is missing or invalid, when it ever needs
more resident Gaussians than the full representation, or when its PSNR in any
experiment falls more than the tolerance below the full representation.
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

from eval.test import render as R

BASELINE = "full_representation"
REQUIRED = ("lodge",)
POOL_FACTOR = 4
EXPERIMENTS = {
    "trajectory_views": {"set": "test_on_path", "kind": "views"},
    "flythrough": {"set": "test_flythrough", "kind": "path"},
    "off_path_views": {"set": "test_off_path", "kind": "views"},
}
QUALITY_TOLERANCE_DB = {"trajectory_views": 1.0, "flythrough": 1.5, "off_path_views": 3.5}
TRANSITION_TOLERANCE_DB = 0.5  # flythrough transition PSNR may not drop more than this below the full representation
VIEW_METRICS = ("psnr", "loaded_gaussians", "visible_gaussians", "ssim", "worst_view_psnr", "peak_loaded_gaussians", "tile_pairs")
PATH_METRICS = ("transition_psnr", "psnr", "ssim", "loaded_gaussians", "visible_gaussians", "peak_loaded_gaussians")
HIGHER_IS_BETTER = {"psnr", "ssim", "worst_view_psnr", "transition_psnr"}


def log(msg: str) -> None:
    print(f"[eval-test {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_hidden(path: Path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: np.array(z[k]) for k in z.files}
    cfg = json.loads(str(data["world_config"]))
    base = {k: data[f"full_{k}"].astype(np.float32) for k in ("means", "scales", "quats", "opacities", "colors")}
    sets = {}
    for exp, spec in EXPERIMENTS.items():
        key = spec["set"]
        cams = R.cameras_from_arrays(data[f"{key}_eye"], data[f"{key}_yaw"], data[f"{key}_pitch"], cfg["focal"], cfg["width"], cfg["height"])
        sets[exp] = (cams, data[f"{key}_images"])
    return cfg, base, sets


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


def evaluate_representation(path: Path, sets: dict, device, max_pool: int) -> dict:
    rep = R.load_representation(path, max_gaussians=max_pool)
    rep_t = R.representation_to_torch(rep, device)
    return {exp: measure(rep, rep_t, *sets[exp], EXPERIMENTS[exp]["kind"]) for exp in EXPERIMENTS}


def discover(checkpoints: Path) -> list[str]:
    names = set(REQUIRED)
    if checkpoints.is_dir():
        for d in sorted(checkpoints.iterdir()):
            if d.is_dir() and not d.name.startswith(("_", ".")) and d.name != BASELINE:
                names.add(d.name)
    return sorted(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--hidden", type=Path, default=Path(__file__).resolve().parent / "hidden_set.npz")
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_test.json"))
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = R.default_device()
    cfg, base, sets = load_hidden(args.hidden)
    n0 = int(base["means"].shape[0])
    max_pool = POOL_FACTOR * n0
    worst_count = float(max_pool)
    log(f"device={device} world={cfg['name']} full representation={n0} Gaussians")

    base_rep = {k: base[k] for k in ("means", "scales", "quats", "opacities", "colors")}
    base_rep.update({
        "band_lo": np.zeros(n0, dtype=np.float32), "band_hi": np.full(n0, np.inf, dtype=np.float32),
        "chunk_centers": np.zeros((1, 3)), "chunks": [np.arange(n0)], "blend_two_nearest": 0,
    })
    base_t = R.representation_to_torch(base_rep, device)
    baseline = {exp: measure(base_rep, base_t, *sets[exp], EXPERIMENTS[exp]["kind"]) for exp in EXPERIMENTS}
    log("baseline: " + json.dumps({e: round(v["psnr"], 3) for e, v in baseline.items()}))

    results = {exp: {BASELINE: {k: baseline[exp][k] for k in metric_names(EXPERIMENTS[exp]["kind"])}} for exp in EXPERIMENTS}
    status = {}
    for name in discover(args.checkpoints_dir):
        path = args.checkpoints_dir / name / "representation.npz"
        reason, measured = None, None
        try:
            measured = evaluate_representation(path, sets, device, max_pool)
            peak = max(measured[e]["peak_loaded_gaussians"] for e in EXPERIMENTS)
            for exp in EXPERIMENTS:
                if not all(math.isfinite(v) for v in measured[exp].values()):
                    reason = f"non-finite metric in {exp}"
            if reason is None and peak > n0:
                reason = f"memory: peak resident Gaussians {peak:.0f} exceed the full representation ({n0})"
            if reason is None:
                for exp in EXPERIMENTS:
                    floor = baseline[exp]["psnr"] - QUALITY_TOLERANCE_DB[exp]
                    if measured[exp]["psnr"] < floor:
                        reason = f"quality: {exp} PSNR {measured[exp]['psnr']:.3f} below floor {floor:.3f}"
                        break
            if reason is None:
                t_floor = baseline["flythrough"]["transition_psnr"] - TRANSITION_TOLERANCE_DB
                if measured["flythrough"]["transition_psnr"] < t_floor:
                    reason = f"transitions: flythrough transition PSNR {measured['flythrough']['transition_psnr']:.3f} below floor {t_floor:.3f}"
        except R.RepresentationError as exc:
            reason = f"invalid representation: {exc}"
        except Exception as exc:  # noqa: BLE001 - any crash is a failed run
            reason = f"evaluation error: {exc.__class__.__name__}: {exc}"
            traceback.print_exc()
        status[name] = {"passed": reason is None, "reason": reason or "ok"}
        if measured is not None:
            status[name]["measured"] = measured
        for exp in EXPERIMENTS:
            kind = EXPERIMENTS[exp]["kind"]
            if reason is None:
                results[exp][name] = {k: measured[exp][k] for k in metric_names(kind)}
            else:
                results[exp][name] = worst_values(kind, worst_count)
        log(f"{name}: {status[name]['reason']}" + ("" if measured is None else " " + json.dumps({e: round(measured[e]['psnr'], 3) for e in EXPERIMENTS})))

    scores = {
        "_meta": {
            "slice": "test",
            "full_representation_gaussians": n0,
            "quality_tolerance_db": QUALITY_TOLERANCE_DB, "transition_tolerance_db": TRANSITION_TOLERANCE_DB,
            "status": status,
        },
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
