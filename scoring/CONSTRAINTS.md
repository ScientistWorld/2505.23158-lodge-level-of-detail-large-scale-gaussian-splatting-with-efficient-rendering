# Scientist Constraints

## Data
- **Allowed inputs:** the public base model, cameras and images in `data/lodge_city/`. Optimise on the `train` views. Rendering the public base model from extra viewpoints is allowed (it is part of the public model).
- **Validation views:** the `val_*` views exist for model selection. Training on them inflates the visible score only; the hidden test uses different cameras.
- **No hidden data:** do not try to obtain or reconstruct the hidden test cameras or targets.

## Output Format
- **What to submit:** write `checkpoints/lodge/representation.npz` in the format documented at the top of `data/gsrender.py`. The evaluator scores every method directory under `checkpoints/` except names starting with `_`; the ranked entry is `lodge`.
- **What the evaluator does:** it chooses resident sets, renders them, and counts loaded and visible Gaussians itself. Nothing else the method writes is read, so do not rely on custom renderers or self-reported numbers.

## Device Budget and Quality Floor (hard gates)
- **Memory:** peak resident Gaussians ≤ 0.65 × the full model (247,095 Gaussians), over every hidden view.
- **Work:** mean visible Gaussians ≤ 0.70 × the full model's, in every experiment.
- **Quality:** PSNR ≥ full model − 1.0 dB (trajectory), − 1.5 dB (flythrough), − 3.5 dB (off-path).
- **Transitions:** flythrough transition PSNR ≥ full model − 0.5 dB.
- **Validity:** at most 4 × the full model Gaussians in the pool, all values finite and in range.

Failing any gate scores the run as failed in every experiment.

## Metrics
Do not substitute wall-clock time for the counted costs. Loaded and visible Gaussian counts are the hardware-independent cost measures this benchmark ranks.

## Compliance Checklist
- [ ] Uses only `data/lodge_city/` (training views for optimisation)
- [ ] Writes `checkpoints/lodge/representation.npz` in the documented format
- [ ] Stays within the memory and work budgets on the visible slice (`bash scripts/evaluate_train.sh`)
- [ ] Keeps PSNR within the quality floors and avoids popping on the visible slice
