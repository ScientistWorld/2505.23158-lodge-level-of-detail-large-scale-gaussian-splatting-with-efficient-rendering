# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-31 13:45] - none
- Read the paper and supplementary details from the markdown.
- Identified the central claim: LOD Gaussian splatting should preserve visual quality while reducing visible and loaded Gaussians.

### [2026-05-31 14:25] - method_runs
- Implemented a compact LODGE reproduction with depth-aware smoothed LOD levels, importance pruning, greedy threshold selection, K-means chunks, visibility filtering, and two-chunk opacity interpolation.
- Added a deterministic synthetic Gaussian-splatting scene and full-representation baseline.
- Smoke-tested the method locally on the scaled scene; it runs end-to-end and writes evaluator artifacts.

### [2026-05-31 14:40] - method_runs
- Audited the inherited reproduction and fixed an evaluation flaw that made the full baseline a perfect copy of the target images.
- Retuned the compact LODGE settings and validated locally that the method runs end-to-end with lower loaded and visible Gaussian counts than the full baseline.
- Submitted a CPU test job to verify the packaged reproduction in the compute environment before advancing to `core_claim`.

### [2026-05-31 14:55] - core_claim
- Compute job `a94396ad-41f` completed successfully inside the packaged container.
- The job reproduced the compact LODGE result: PSNR 26.805 and SSIM 0.864 while reducing loaded Gaussians from 848.0 to 593.6 and visible Gaussians from 581.2 to 561.2 versus the full representation.
- `python validate.py --compare` passed in the compute environment, confirming the evaluator consumes generated artifacts and matches the reference schema.

## Stop Justification
- Completed at milestone `core_claim`.
