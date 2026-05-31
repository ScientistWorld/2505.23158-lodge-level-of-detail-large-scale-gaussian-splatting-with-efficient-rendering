# Progress

## What Works

Implemented a compact CPU-scale reproduction of LODGE. The pipeline builds lower-detail Gaussian levels from a full Gaussian representation, selects thresholds greedily from a tile-cost proxy, clusters camera positions into chunks, filters chunk visibility by importance, and renders the two nearest chunks with opacity interpolation.

The baseline full representation and the LODGE method both write `renders.npz` and `stats.json` artifacts. `scripts/evaluate.sh` reads those artifacts without importing `method/` and writes `scoring/scores.json`.

## Results

Local smoke-test results before compute submission showed the method running end-to-end on the compact synthetic scene. With the current tuned settings, LODGE reduced loaded Gaussians and tile/visible work substantially relative to the full representation, while retaining non-trivial held-out view quality. Final authoritative numbers should come from the submitted job.

## Remaining

Run the packaged reproduction on a compute node and inspect `scoring/scores.json`. If the submitted job confirms the local smoke result, the milestone can be advanced to `core_claim`. Higher milestones would require adding more settings, such as a larger scene, a manual-threshold ablation, and a no-blending chunk-boundary ablation.

## Issues

The full paper benchmarks require large 3DGS scene assets and long CUDA training/evaluation runs. This reproduction uses a compact deterministic synthetic scene to preserve the computational procedure without shipping multi-GB data.

## Deviations from Paper

The paper evaluates on Hierarchical 3DGS and Zip-NeRF scenes with 1000-2000 images and trained 3DGS representations. This environment uses a deterministic synthetic Gaussian-splatting scene with small images and hundreds of primitives so the benchmark can run quickly and portably.

The paper fine-tunes each LOD level with 1000 gradient steps after each pruning round. This compact reproduction keeps the same prune/fine-tune stages but uses deterministic visibility-statistic parameter updates instead of CUDA gradient optimization, preserving the stage and affected parameters while avoiding heavy login-node or test-node compute.

The paper reports FPS; this benchmark scores loaded Gaussians, visible Gaussians, and tile cost instead because wall-clock speed is hardware dependent.
