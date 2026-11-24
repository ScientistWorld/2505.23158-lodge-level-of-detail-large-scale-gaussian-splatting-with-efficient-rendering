# Verification (iteration 11)

- **`python validate.py --compare --post-design`:** all checks pass. The one heuristic warning is on `eval/test/world.py`, the hidden world generator; it is not an evaluator.
- **Rasteriser:** a self-check against a per-tile loop gives a maximum difference of 2.4e-7. It runs at the start of every audit job.
- **Hidden test:** every case was scored with the unchanged `scripts/evaluate_test.sh` in a fresh process. Per-case outputs are in `notes/hardening/cases/`.
- **Seed spread** of the reference (3 seeds), hidden test:

  | Metric | Spread |
  |---|---:|
  | Trajectory PSNR | 0.010 dB |
  | Flythrough PSNR | 0.083 dB |
  | Off-path PSNR | 0.006 dB |
  | Loaded Gaussians | 134 |
  | Visible Gaussians | 43 |

- **Train/test separation:** `eval/train` and `eval/test` share no code. `eval/` never imports `method/`. The hidden test reads only `checkpoints/*/representation.npz` and files inside `eval/test/`.
