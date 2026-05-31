"""Full-representation Gaussian splatting baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.gaussian_scene import ensure_dataset, load_dataset, render_scene, stack_renders


def scalar_stats(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if k != "importance"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/lodge_synthetic"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/full_representation"))
    args = parser.parse_args()

    ensure_dataset(args.data_dir)
    scene, _, test, _ = load_dataset(args.data_dir)
    renders = []
    stats = []
    for cam in test:
        img, st = render_scene(scene, cam, return_stats=True)
        renders.append(img)
        stats.append(scalar_stats(st))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "renders.npz", images=stack_renders(renders))
    (args.out_dir / "stats.json").write_text(
        json.dumps(
            {
                "config": {"method": "full_representation", "level_gaussians": [int(scene["means"].shape[0])]},
                "per_view": stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
