# Evaluation

## Protocol

The evaluator loads `representation.npz` for every method directory under `checkpoints/`. For each camera it:

1. finds the resident set (nearest chunk, or the union of the two nearest with the format's opacity interpolation; distance bands applied);
2. renders it with a fixed tile-based 3DGS rasteriser (EWA projection, Mip-Splatting 2D filter, 16×16 tiles, alpha in [1/255, 0.99], early termination);
3. quantises the render to uint8 and compares it with the uint8 target.

Nothing the method writes besides the representation is read. The full-representation baseline is rendered by the evaluator from its own copy of the base model.

## Experiments

| Experiment | Cameras | Scored metrics (coefficient) | Weight |
|---|---|---|---|
| `trajectory_views` | held-out views interleaved along the capture path (every 8th frame) | psnr (1.0), loaded_gaussians (0.35), visible_gaussians (0.35) | 0.45 |
| `flythrough` | a dense forward-moving path through the street | psnr (1.0), transition_psnr (0.25) | 0.20 |
| `off_path_views` | views off the capture path: lateral offsets, heights from 1.2 to 4.5 m, any heading | psnr (1.0), loaded_gaussians (0.35), visible_gaussians (0.35) | 0.35 |

### Metric definitions

- **psnr**: mean PSNR in dB over views, uint8 protocol.
- **loaded_gaussians**: mean size of the resident set per view (memory).
- **visible_gaussians**: mean number of resident Gaussians that the rasteriser processes per view. These are in front of the near plane, have a 3σ footprint overlapping the image, and have opacity × weight ≥ 1/255.
- **transition_psnr**: temporal PSNR of the 5 worst frame-to-frame changes along the path. It compares the rendered change with the target change, so popping lowers it.
- **Diagnostics** (coefficient 0): ssim, worst_view_psnr, peak_loaded_gaussians, tile_pairs.

## Gates

If any gate fails, the method gets the worst value of every metric in every experiment (PSNR 0, counts 4× the full model).

- **Validity:** the representation must be readable and valid:
  - all values finite;
  - scales in (1e-7, 100] m;
  - opacities and colours in [0, 1];
  - CSR chunk lists with in-range, duplicate-free indices;
  - pool at most 4× the full model.
- **Memory budget:** peak resident Gaussians over all hidden views at most 0.65× the full model.
- **Work budget:** mean visible Gaussians at most 0.70× the full representation's, in every experiment.
- **Quality floor:** PSNR at least the full representation's minus 1.0 dB (trajectory), 1.5 dB (flythrough) and 3.5 dB (off-path).
- **Transitions:** flythrough transition_psnr at least the full representation's minus 0.5 dB.

## Slices

`scripts/evaluate_train.sh` (and `scripts/evaluate.sh`) score the public validation cameras in `data/lodge_city`, using the same protocol and gates. `scripts/evaluate_test.sh` (no arguments) scores the hidden cameras and writes `scoring/scores_test.json`.
