# Train/Test Evaluation Design Notes

## What I Checked

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. They invoke `eval/train/evaluate.py` and `eval/test/evaluate.py` respectively, and write `scoring/scores_train.json` / `scoring/scores_test.json`.
- Checked that the slice evaluators read saved method artifacts from `checkpoints/*/renders.npz` and `checkpoints/*/stats.json` rather than importing or rerunning `method/`.
- Grepped for cross-imports between `eval/train/` and `eval/test/`; none were present beyond path strings in scripts and docstrings.
- Checked `scoring/reference.json` for paper-agreement metric names or formulas. The metrics score scientific quantities directly: quality and machine-independent cost proxies.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.
- Ran `bash -n` on both train/test evaluation scripts. I did not run the evaluators or any experiments in this design-only iteration.

## What I Changed

- Fixed the method-discovery weakness in the train/test evaluators. The previous pass only scored method names listed in `scoring/reference.json`, so a future method written to a new `checkpoints/<name>/` directory could be silently ignored. Both slices now discover every checkpoint directory with the artifact contract and include those methods. The test slice still follows `scores_train.json` method keys when train scores exist, preserving train/test key parity for final scoring.
- Added metric triangulation to both train and test slices:
  - `worst_view_psnr` catches methods that optimize average quality while failing hard on some views.
  - `ssim` adds a structural-fidelity signal beyond MSE/PSNR.
  - `peak_loaded_gaussians` and `peak_visible_gaussians` catch cost spikes hidden by averages.
  - `tile_cost` adds a hardware-independent overdraw proxy tied to rasterization pressure.
- Updated `scoring/reference.json` so every nonzero metric has `higher_is_better`, `coefficient`, and `description`, and so the metric contract matches what the slice evaluators emit.

## Runtime Concerns For The Runner Iteration

- The camera pool is small: the visible slice uses three held-out test views and the held-out slice uses two. The added worst-view and peak-cost metrics reduce average-metric gaming, but the next runtime pass should watch for saturation or excessive variance from the small split.
- `tile_cost` is now required in `stats.json`; this avoids rewarding incomplete artifacts with an artificial zero-cost score. The next runtime pass should confirm all current and future-facing artifact writers include it.
- I did not execute the Python evaluators, so the next runner iteration should verify that the expanded metric keys appear in generated `scores_train.json` and `scores_test.json` and that train/test values are not accidentally identical.
