"""Write the base reconstruction (full representation) in the submission format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.gsrender import load_base_gaussians, load_camera_sets, save_representation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/lodge_city"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/_examples/full_representation/representation.npz"))
    args = parser.parse_args()
    base = load_base_gaussians(args.data_dir)
    train = load_camera_sets(args.data_dir)["train"]
    center = np.stack([c.eye for c in train]).mean(0)
    n = base["means"].shape[0]
    save_representation(args.out, base, center[None], [np.arange(n)], blend_two_nearest=False)
    print(f"wrote {args.out} ({n} Gaussians, one chunk)")


if __name__ == "__main__":
    main()
