# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-31 13:55] - method_runs
- Continuing audit reset the active milestone to `method_runs`.
- Existing core comparison is structurally sound, but this turn adds chunk/blending ablations that must run in the packaged environment before claiming a higher milestone.

### [2026-05-31 13:41] - core_claim
- Implemented and packaged a compact LODGE reproduction with depth-aware LOD smoothing, importance pruning, automatic threshold selection, chunked active-Gaussian caching, visibility filtering, and two-chunk opacity blending.
- Ran the full-representation baseline and LODGE method through the method-independent evaluator.
- Reproduced the central quality-efficiency claim on the compact synthetic Gaussian-splatting benchmark: LODGE preserved held-out rendering quality close to the full representation while reducing loaded and visible Gaussian counts.

## Stop Justification
- Not stopped; submitting a packaged-environment job for the expanded ablation suite.
