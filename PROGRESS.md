# Progress

## What Works

Implemented a compact CPU-scale reproduction of LODGE. The pipeline builds lower-detail Gaussian levels from a full Gaussian representation, selects thresholds greedily from a tile-cost proxy, clusters camera positions into chunks, filters chunk visibility by importance, and renders the two nearest chunks with opacity interpolation. LOD construction now uses a small autograd fine-tuning stage with the paper's L1+DSSIM loss after each pruning round, replacing the earlier visibility-statistic substitute.

The baseline full representation and the LODGE method both write `renders.npz` and `stats.json` artifacts. `scripts/evaluate.sh` reads those artifacts without importing `method/` and writes `scoring/scores.json`.

## Results

Local smoke-test results before compute submission showed the method running end-to-end on the compact synthetic scene. With the current tuned settings, LODGE reduced loaded and visible Gaussians relative to the full representation, while retaining held-out view quality close to the full baseline.

Latest packaged-container `bash scripts/reproduce.sh` result from job `2c129847-7c6`:

| Method | PSNR | SSIM | Loaded Gaussians | Visible Gaussians |
| --- | ---: | ---: | ---: | ---: |
| full_representation | 27.329 | 0.899 | 848.0 | 581.2 |
| lodge | 26.805 | 0.864 | 593.6 | 561.2 |
| lod_dynamic | 27.973 | 0.899 | 705.2 | 584.6 |
| nearest_chunk | 26.283 | 0.850 | 507.4 | 491.0 |
| opacity_blend | 26.805 | 0.864 | 593.6 | 561.2 |

`python validate.py --compare` passes on these artifacts. Compute job `2c129847-7c6` reproduced the core comparison and the added ablation in the packaged container. This supports `secondary_claims`: the paper's LOD/chunking computation lowers memory/work proxies while maintaining rendering quality close to the full representation, and opacity blending improves chunk-transition quality over nearest-chunk rendering at a moderate cost increase.

Current iteration: replaced the compact LOD fine-tuning approximation with actual gradient optimization and removed SSIM from the paper-reference scoring contract because the reproduced SmallCity Table 3 ablation reports PSNR and Gaussian counts, not SSIM. A new CPU compute job is being submitted to regenerate artifacts with the stricter method implementation.

## Remaining

The reusable benchmark is complete through the main claim and one design-validation ablation. Higher milestones would require substantially larger scene coverage or more of the paper's full benchmark tables, which would need larger 3DGS assets and longer CUDA runs.

## Issues

The full paper benchmarks require large 3DGS scene assets and long CUDA training/evaluation runs. This reproduction uses a compact deterministic synthetic scene to preserve the computational procedure without shipping multi-GB data.

An audit found that an earlier synthetic target was generated directly from the base full representation, which made the full baseline artificially perfect. The dataset generator now mixes the base representation with a denser deterministic reference scene so both baseline and LODGE are scored against finite held-out targets. The audit also removed the paper's wall-clock runtime column from `reference.json`; official constraints now use the paper-reported machine-independent visible and loaded Gaussian counts.

## Deviations from Paper

The paper evaluates on Hierarchical 3DGS and Zip-NeRF scenes with 1000-2000 images and trained 3DGS representations. This environment uses a deterministic synthetic Gaussian-splatting scene with small images and hundreds of primitives so the benchmark can run quickly and portably.

The paper fine-tunes each LOD level with 1000 gradient steps after each pruning round. This compact reproduction scales the iteration count down for the small CPU benchmark, but keeps a real gradient-optimization stage on the surviving LOD Gaussian scale, opacity, and color parameters with the same L1+DSSIM loss family.

The paper reports FPS; this benchmark scores loaded Gaussians and visible Gaussians instead because wall-clock speed is hardware dependent.
