# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-31 13:56] - core_claim_plus
- Audited the available job logs and found they predate the stricter autograd fine-tuning implementation and current scoring contract.
- Updated `scripts/run.sh` to clear generated checkpoints, synthetic data, and stale scores before rerunning the full reproduction.
- Resubmitting a CPU reproduction job to verify the current implementation in the packaged environment.

### [2026-05-31 13:52] - core_claim_plus
- Replaced the compact LOD fine-tuning approximation with a real autograd optimization stage using the paper's L1+DSSIM loss family.
- Removed non-Table-3 SSIM values from `reference.json` and the scoring contract so reference metrics are paper-reported.
- Submitting a CPU reproduction job to regenerate baseline, LODGE, and chunk-ablation artifacts with the updated implementation.

## Stop Justification
- Not stopped; continuing with stricter method validation.
