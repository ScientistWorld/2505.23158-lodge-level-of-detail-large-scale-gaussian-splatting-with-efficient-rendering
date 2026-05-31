"""Score the visible train slice without importing method code.

This evaluator reads rendered artifacts produced by a method under
``checkpoints/`` and computes the same scientific quantities as the original
benchmark on a deterministic subset of held-out camera views.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.gaussian_scene import ensure_dataset, load_dataset, psnr, simple_ssim


SLICE_NAME = "train"
VIEW_INDICES = (0, 2, 4)
EXPERIMENTS = (
    "synthetic_lod_quality_efficiency",
    "synthetic_chunk_blending_ablation",
)
ALIASED_ARTIFACTS = {"opacity_blend": "lodge"}


def uint8_round(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x, 0, 1) * 255.0).astype(np.uint8).astype(np.float32) / 255.0


def load_reference_methods(reference_path: Path) -> dict[str, list[str]]:
    reference = json.loads(reference_path.read_text())
    experiments = reference.get("experiments", {})
    return {
        exp_name: list(exp.get("results", {}).keys())
        for exp_name, exp in experiments.items()
        if exp_name in EXPERIMENTS
    }


def discover_artifact_methods(checkpoints_dir: Path) -> list[str]:
    if not checkpoints_dir.exists():
        return []
    return sorted(
        child.name
        for child in checkpoints_dir.iterdir()
        if child.is_dir()
        and (child / "renders.npz").exists()
        and (child / "stats.json").exists()
    )


def merge_methods(reference_methods: list[str], artifact_methods: list[str]) -> list[str]:
    methods = list(reference_methods)
    aliased_artifacts = {
        artifact_name
        for alias_name, artifact_name in ALIASED_ARTIFACTS.items()
        if alias_name in methods
    }
    for name in artifact_methods:
        if name in aliased_artifacts:
            continue
        if name not in methods:
            methods.append(name)
    return methods


def resolve_run_dir(method_name: str, checkpoints_dir: Path) -> Path | None:
    direct = checkpoints_dir / method_name
    if (direct / "renders.npz").exists() and (direct / "stats.json").exists():
        return direct

    # The original ablation reports the LODGE run as the opacity-blended
    # variant. Prefer an explicit directory if future methods provide one.
    if method_name == "opacity_blend":
        lodge = checkpoints_dir / "lodge"
        if (lodge / "renders.npz").exists() and (lodge / "stats.json").exists():
            return lodge

    return None


def evaluate_run(run_dir: Path, data_dir: Path, view_indices: tuple[int, ...]) -> dict[str, float]:
    ensure_dataset(data_dir)
    _, _, _, gt = load_dataset(data_dir)
    pred = np.load(run_dir / "renders.npz")["images"]
    meta = json.loads((run_dir / "stats.json").read_text())
    stats = meta["per_view"]

    max_index = max(view_indices)
    if pred.shape[0] <= max_index or gt.shape[0] <= max_index or len(stats) <= max_index:
        raise ValueError(
            f"{run_dir} does not contain enough per-view outputs for {SLICE_NAME} slice "
            f"indices {list(view_indices)}"
        )

    pred_q = uint8_round(pred)
    gt_q = uint8_round(gt)
    psnrs = [psnr(pred_q[i], gt_q[i]) for i in view_indices]
    ssims = [simple_ssim(pred_q[i], gt_q[i]) for i in view_indices]
    loaded = [float(stats[i]["loaded"]) for i in view_indices]
    visible = [float(stats[i]["visible"]) for i in view_indices]
    tile_cost = [float(stats[i]["tile_cost"]) for i in view_indices]
    return {
        "psnr": float(np.mean(psnrs)),
        "worst_view_psnr": float(np.min(psnrs)),
        "ssim": float(np.mean(ssims)),
        "loaded_gaussians": float(np.mean(loaded)),
        "visible_gaussians": float(np.mean(visible)),
        "peak_loaded_gaussians": float(np.max(loaded)),
        "peak_visible_gaussians": float(np.max(visible)),
        "tile_cost": float(np.mean(tile_cost)),
    }


def build_scores(args: argparse.Namespace) -> dict:
    methods_by_experiment = load_reference_methods(args.reference)
    artifact_methods = discover_artifact_methods(args.checkpoints_dir)
    experiments: dict[str, dict] = {}

    for exp_name in EXPERIMENTS:
        results = {}
        method_names = merge_methods(methods_by_experiment.get(exp_name, []), artifact_methods)
        for method_name in method_names:
            run_dir = resolve_run_dir(method_name, args.checkpoints_dir)
            if run_dir is None:
                continue
            results[method_name] = evaluate_run(run_dir, args.data_dir, VIEW_INDICES)
        experiments[exp_name] = {
            "_meta": {
                "slice": SLICE_NAME,
                "view_indices": list(VIEW_INDICES),
            },
            "results": results,
        }

    return {
        "_meta": {
            "slice": SLICE_NAME,
            "data_pool": "lodge_synthetic/test_gt",
            "split_rule": "fixed held-out camera subset",
        },
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/lodge_synthetic"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--reference", type=Path, default=Path("scoring/reference.json"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    scores = build_scores(args)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
