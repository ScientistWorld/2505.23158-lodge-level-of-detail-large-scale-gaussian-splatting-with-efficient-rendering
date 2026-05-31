"""Evaluate rendered outputs without importing the method implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.gaussian_scene import ensure_dataset, load_dataset, psnr, simple_ssim


def uint8_round(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x, 0, 1) * 255.0).astype(np.uint8).astype(np.float32) / 255.0


def evaluate_run(run_dir: Path, data_dir: Path) -> dict:
    ensure_dataset(data_dir)
    _, _, _, gt = load_dataset(data_dir)
    pred = np.load(run_dir / "renders.npz")["images"]
    meta = json.loads((run_dir / "stats.json").read_text())
    pred_q = uint8_round(pred)
    gt_q = uint8_round(gt)
    stats = meta["per_view"]
    psnrs = [psnr(p, g) for p, g in zip(pred_q, gt_q)]
    ssims = [simple_ssim(p, g) for p, g in zip(pred_q, gt_q)]
    loaded = [float(s["loaded"]) for s in stats]
    visible = [float(s["visible"]) for s in stats]
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "loaded_gaussians": float(np.mean(loaded)),
        "visible_gaussians": float(np.mean(visible)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/lodge_synthetic"))
    parser.add_argument("--full-dir", type=Path, default=Path("checkpoints/full_representation"))
    parser.add_argument("--lodge-dir", type=Path, default=Path("checkpoints/lodge"))
    parser.add_argument("--lod-dynamic-dir", type=Path, default=Path("checkpoints/lod_dynamic"))
    parser.add_argument("--nearest-chunk-dir", type=Path, default=Path("checkpoints/nearest_chunk"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()

    main_results = {}
    if args.full_dir.exists():
        main_results["full_representation"] = evaluate_run(args.full_dir, args.data_dir)
    if args.lodge_dir.exists():
        main_results["lodge"] = evaluate_run(args.lodge_dir, args.data_dir)

    ablation_results = {}
    if args.lod_dynamic_dir.exists():
        ablation_results["lod_dynamic"] = evaluate_run(args.lod_dynamic_dir, args.data_dir)
    if args.nearest_chunk_dir.exists():
        ablation_results["nearest_chunk"] = evaluate_run(args.nearest_chunk_dir, args.data_dir)
    if args.lodge_dir.exists():
        ablation_results["opacity_blend"] = evaluate_run(args.lodge_dir, args.data_dir)

    experiments = {"synthetic_lod_quality_efficiency": {"results": main_results}}
    if ablation_results:
        experiments["synthetic_chunk_blending_ablation"] = {"results": ablation_results}
    scores = {"experiments": experiments}
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
