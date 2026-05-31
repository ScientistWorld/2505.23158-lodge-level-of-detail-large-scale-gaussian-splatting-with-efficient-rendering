# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-31 14:42] - method_runs
- Audited the existing scaffold against the paper and found `secondary_claims` was overstated.
- Patched the method toward the paper-specific computation: LOD-stack fine-tuning with randomized current threshold, nearest-center chunk radius, random-orientation visibility filtering, and Eq. (4)-style opacity blending.
- Submitting a clean CPU reproduction job to regenerate scores before claiming `core_claim` again.

### [2026-05-31 14:36] - secondary_claims
- Final verification completed.
- Expanded `scripts/evaluate.sh` metrics to match the full scoring contract and updated `scoring/scores.json`.
- Cleaned method-agnostic experiment documentation and removed tracked generated job/trajectory artifacts.
- `python validate.py --compare --post-design` passes.

### [2026-05-31 14:34] - secondary_claims
- Re-audited the current workspace as the reproducer handoff state.
- `python validate.py --compare --post-design` passes, including reference/scores validation, import separation, train/test independence, artifact-based evaluation, and split-score differentiation.
- Leaving milestone at `secondary_claims`: the packaged compact benchmark reproduces the core LODGE quality/efficiency trade-off and chunk-blending ablation.

### [2026-05-31 14:14] - secondary_claims
- CPU job `f840ed37-d90` completed successfully with the current autograd fine-tuning implementation.
- `python validate.py --compare` passed in the packaged container, including import-separation and artifact-based evaluation checks.
- Restored `secondary_claims`: the main quality/efficiency claim and the chunk-caching/blending ablation are both reproduced on the compact deterministic scene.

### [2026-05-31 13:56] - core_claim_plus
- Audited the available job logs and found they predate the stricter autograd fine-tuning implementation and current scoring contract.
- Updated `scripts/run.sh` to clear generated checkpoints, synthetic data, and stale scores before rerunning the full reproduction.
- Resubmitting a CPU reproduction job to verify the current implementation in the packaged environment.

### [2026-05-31 13:52] - core_claim_plus
- Replaced the compact LOD fine-tuning approximation with a real autograd optimization stage using the paper's L1+DSSIM loss family.
- Removed non-Table-3 SSIM values from `reference.json` and the scoring contract so reference metrics are paper-reported.
- Submitting a CPU reproduction job to regenerate baseline, LODGE, and chunk-ablation artifacts with the updated implementation.

## Stop Justification
- Not stopped; current patched method is pending compute-job validation.
