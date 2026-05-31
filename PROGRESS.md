# Progress

## What Works

Implemented a compact CPU-scale reproduction of LODGE. The pipeline builds lower-detail Gaussian levels from a full Gaussian representation, selects thresholds greedily from a tile-cost proxy, clusters camera positions into chunks, filters chunk visibility by importance, and renders the two nearest chunks with opacity interpolation.

The baseline full representation and the LODGE method both write `renders.npz` and `stats.json` artifacts. `scripts/evaluate.sh` reads those artifacts without importing `method/` and writes `scoring/scores.json`.

## Results

Local smoke-test results before compute submission showed the method running end-to-end on the compact synthetic scene. With the current tuned settings, LODGE reduced loaded and visible Gaussians relative to the full representation, while retaining held-out view quality close to the full baseline.

Latest local `bash scripts/reproduce.sh` result:

| Method | PSNR | SSIM | Loaded Gaussians | Visible Gaussians |
| --- | ---: | ---: | ---: | ---: |
| full_representation | 27.329 | 0.899 | 848.0 | 581.2 |
| lodge | 26.805 | 0.864 | 593.6 | 561.2 |

`python validate.py --compare` passes on these artifacts. Compute job `a94396ad-41f` reproduced the same numbers in the packaged container and exited successfully.

## Remaining

The reusable core benchmark is complete. Higher milestones would require adding more settings, such as a larger scene, a manual-threshold ablation, and a no-blending chunk-boundary ablation.

## Issues

The full paper benchmarks require large 3DGS scene assets and long CUDA training/evaluation runs. This reproduction uses a compact deterministic synthetic scene to preserve the computational procedure without shipping multi-GB data.

An audit found that an earlier synthetic target was generated directly from the base full representation, which made the full baseline artificially perfect. The dataset generator now mixes the base representation with a denser deterministic reference scene so both baseline and LODGE are scored against finite held-out targets. The audit also removed the paper's wall-clock runtime column from `reference.json`; official constraints now use the paper-reported machine-independent visible and loaded Gaussian counts.

## Deviations from Paper

The paper evaluates on Hierarchical 3DGS and Zip-NeRF scenes with 1000-2000 images and trained 3DGS representations. This environment uses a deterministic synthetic Gaussian-splatting scene with small images and hundreds of primitives so the benchmark can run quickly and portably.

The paper fine-tunes each LOD level with 1000 gradient steps after each pruning round. This compact reproduction keeps the same prune/fine-tune stages but uses deterministic visibility-statistic parameter updates instead of CUDA gradient optimization, preserving the stage and affected parameters while avoiding heavy login-node or test-node compute.

The paper reports FPS; this benchmark scores loaded Gaussians and visible Gaussians instead because wall-clock speed is hardware dependent.
