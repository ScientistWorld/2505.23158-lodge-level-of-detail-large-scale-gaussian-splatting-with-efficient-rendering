"""Time the renderer on the real world data (diagnostic for job planning)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data import gsrender as R


def timeit(fn, reps=5):
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps, out


def main():
    dev = R.default_device()
    data = ROOT / "data" / "lodge_city"
    base = {k: torch.as_tensor(v, device=dev) for k, v in R.load_base_gaussians(data).items()}
    cams = R.load_camera_sets(data)["train"]
    print("N", base["means"].shape[0], "device", dev, torch.cuda.get_device_name(0))
    for i in (0, 1, 2, 30, 60, 90):
        cam = cams[i]
        with torch.no_grad():
            dt, out = timeit(lambda: R.render_gaussians(base["means"], base["scales"], base["quats"], base["opacities"], base["colors"], cam))
        print(f"view {i} yaw {cam.yaw:.2f}: no-grad {dt*1000:.1f} ms visible {out['visible']} pairs {out['tile_pairs']}")
    cam = cams[0]
    for mp in (1 << 15, 1 << 17, 1 << 19):
        R.MAX_PAIRS_PER_GROUP = mp
        with torch.no_grad():
            dt, _ = timeit(lambda: R.render_gaussians(base["means"], base["scales"], base["quats"], base["opacities"], base["colors"], cam))
        print(f"max_pairs {mp}: no-grad {dt*1000:.1f} ms")
    R.MAX_PAIRS_PER_GROUP = 1 << 17
    op = base["opacities"].clone().requires_grad_(True)

    def fb():
        out = R.render_gaussians(base["means"], base["scales"], base["quats"], op, base["colors"], cam)
        out["image"].mean().backward()
        return out

    dt, _ = timeit(fb, reps=3)
    print(f"forward+backward {dt*1000:.1f} ms")
    with torch.no_grad():
        dt, _ = timeit(lambda: R.render_gaussians(base["means"], base["scales"], base["quats"], base["opacities"], base["colors"], cam, return_importance=True))
    print(f"no-grad with importance {dt*1000:.1f} ms")
    torch.use_deterministic_algorithms(True, warn_only=True)
    dt, _ = timeit(fb, reps=3)
    print(f"forward+backward deterministic {dt*1000:.1f} ms")
    torch.use_deterministic_algorithms(False)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fb()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12))


if __name__ == "__main__":
    main()
