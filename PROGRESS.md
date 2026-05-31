# Progress

## What Works

Implemented a compact CPU-scale reproduction of LODGE. The pipeline builds lower-detail Gaussian levels from a full Gaussian representation, selects thresholds greedily from a tile-cost proxy, clusters camera positions into chunks, filters chunk visibility by importance, and renders the two nearest chunks with opacity interpolation. LOD construction now uses a small autograd fine-tuning stage with the paper's L1+DSSIM loss after each pruning round, replacing the earlier visibility-statistic substitute.

The baseline full representation and the LODGE method both write `renders.npz` and `stats.json` artifacts. `scripts/evaluate.sh` reads those artifacts without importing `method/` and writes `scoring/scores.json`.

## Results

Packaged-container job `f840ed37-d90` reran `scripts/run.sh` after clearing generated data, checkpoints, and stale scores. It completed with exit code 0 and `python validate.py --compare` passed, including structural checks that evaluation reads method artifacts and does not import from `method/`.

Final local audit on 2026-05-31 14:34 UTC ran `python validate.py --compare --post-design`. It passed all checks, including split train/test score validation, import separation, no `shared/` references in committed code, and method-key consistency across train and test scores.

Latest verified `bash scripts/reproduce.sh` results:

| Method | PSNR | Loaded Gaussians | Visible Gaussians |
| --- | ---: | ---: | ---: |
| full_representation | 27.329 | 848.0 | 581.2 |
| lodge | 27.456 | 591.6 | 560.2 |
| lod_dynamic | 28.375 | 705.2 | 584.0 |
| nearest_chunk | 26.928 | 505.2 | 491.6 |
| opacity_blend | 27.456 | 591.6 | 560.2 |

This supports `secondary_claims`: the paper's LOD/chunking computation lowers loaded-Gaussian and visible-Gaussian proxies while maintaining held-out rendering quality, and the chunk-caching/blending ablation exposes the expected quality/cost trade-off. The current scoring contract uses PSNR plus machine-independent Gaussian counts because the reproduced SmallCity Table 3 ablation reports PSNR and Gaussian counts, not SSIM.

## Remaining

The reusable benchmark is complete through the main claim and one design-validation ablation. Higher milestones would require substantially larger scene coverage or more of the paper's full benchmark tables, which would need larger 3DGS assets and longer CUDA runs.

## Issues

The full paper benchmarks require large 3DGS scene assets and long CUDA training/evaluation runs. This reproduction uses a compact deterministic synthetic scene to preserve the computational procedure without shipping multi-GB data.

An audit found that an earlier synthetic target was generated directly from the base full representation, which made the full baseline artificially perfect. The dataset generator now mixes the base representation with a denser deterministic reference scene so both baseline and LODGE are scored against finite held-out targets. The audit also removed the paper's wall-clock runtime column from `reference.json`; official constraints now use the paper-reported machine-independent visible and loaded Gaussian counts.

## Deviations from Paper

The paper evaluates on Hierarchical 3DGS and Zip-NeRF scenes with 1000-2000 images and trained 3DGS representations. This environment uses a deterministic synthetic Gaussian-splatting scene with small images and hundreds of primitives so the benchmark can run quickly and portably.

The paper fine-tunes each LOD level with 1000 gradient steps after each pruning round. This compact reproduction scales the iteration count down for the small CPU benchmark, but keeps a real gradient-optimization stage on the surviving LOD Gaussian scale, opacity, and color parameters with the same L1+DSSIM loss family.

The paper reports FPS; this benchmark scores loaded Gaussians and visible Gaussians instead because wall-clock speed is hardware dependent.
