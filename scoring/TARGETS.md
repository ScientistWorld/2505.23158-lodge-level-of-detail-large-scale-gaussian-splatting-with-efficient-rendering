# Evaluation Targets

## Primary: trajectory_views
Held-out capture-trajectory views: quality close to the full model with a much smaller resident and processed set.
- **Metric**: psnr (higher is better)
- **How to run**: `bash scripts/method.sh` then `bash scripts/evaluate_test.sh`

## Constraint: trajectory_views
Memory footprint and rasterisation work; also gated by the device budget (peak resident ≤ 0.65× full model, mean visible ≤ 0.70× full model) and the quality floor (full model − 1.0 dB).
- **Metric**: loaded_gaussians (lower is better)
- **Metric**: visible_gaussians (lower is better)
- **How to run**: `bash scripts/evaluate_test.sh`

## Secondary: flythrough
Dense camera path: no popping when resident sets change, quality along the path.
- **Metric**: psnr (higher is better)
- **Metric**: transition_psnr (higher is better; gate: ≥ full model − 0.5 dB)
- **How to run**: `bash scripts/evaluate_test.sh`

## Secondary (headroom): off_path_views
Views away from the capture trajectory; quality floor is the full model − 3.5 dB.
- **Metric**: psnr (higher is better)
- **Metric**: loaded_gaussians (lower is better)
- **Metric**: visible_gaussians (lower is better)
- **How to run**: `bash scripts/evaluate_test.sh`
