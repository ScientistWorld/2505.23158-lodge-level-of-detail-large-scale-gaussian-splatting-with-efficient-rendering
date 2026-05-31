# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

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
- Stopping at `secondary_claims`: the core LODGE quality/efficiency result and the Table-3-style chunk design validation are verified in the packaged environment. Larger paper coverage would require large 3DGS scene assets and longer CUDA-scale experiments beyond this compact research gym.
