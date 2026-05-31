# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-05-31 13:56] - secondary_claims
- Compute job `2c129847-7c6` ran successfully in the packaged container.
- Restored the core quality-efficiency claim and added a scored ablation for dynamic LOD, nearest cached chunks, and two-chunk opacity blending.
- The ablation reproduces the paper's secondary design claim qualitatively: cached chunks lower loaded/visible Gaussian counts, and opacity blending recovers quality compared with nearest-chunk rendering while still using fewer Gaussians than the full representation.

### [2026-05-31 13:55] - method_runs
- Continuing audit reset the active milestone to `method_runs`.
- Existing core comparison is structurally sound, but this turn adds chunk/blending ablations that must run in the packaged environment before claiming a higher milestone.

### [2026-05-31 13:41] - core_claim
- Implemented and packaged a compact LODGE reproduction with depth-aware LOD smoothing, importance pruning, automatic threshold selection, chunked active-Gaussian caching, visibility filtering, and two-chunk opacity blending.
- Ran the full-representation baseline and LODGE method through the method-independent evaluator.
- Reproduced the central quality-efficiency claim on the compact synthetic Gaussian-splatting benchmark: LODGE preserved held-out rendering quality close to the full representation while reducing loaded and visible Gaussian counts.

## Stop Justification
- Completed at milestone `secondary_claims`.
