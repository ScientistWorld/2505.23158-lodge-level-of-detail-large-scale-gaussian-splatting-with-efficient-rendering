# Evaluation

## Metrics

The compact benchmark reports PSNR and a simple SSIM implementation on held-out rendered RGB images after rounding predictions and references to uint8, matching the paper's metric protocol. Higher PSNR and SSIM are better.

Efficiency is measured by the average number of loaded Gaussians, average number of visible Gaussians, and average 16x16 tile cost over test views. Lower values are better. These are machine-independent proxies for memory use and rasterization work, replacing wall-clock FPS in the reusable benchmark.

## Baselines and Targets

The baseline is the full Gaussian representation rendered without any memory- or detail-reduction scheme. The target is to preserve image quality while reducing loaded and visible Gaussians by a large factor. `scoring/reference.json` includes the paper's SmallCity Table 3 values for context and the compact experiment structure used by this environment.

## Evaluation Protocol

Run `bash scripts/reproduce.sh` to generate deterministic test renders for the full baseline and proposed method, then compute scores with `bash scripts/evaluate.sh`. The evaluator reads only output artifacts under `checkpoints/` and writes `scoring/scores.json`; it does not import or rerun `method/`.
