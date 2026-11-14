# Static findings pending job evidence

These are code-inspection findings. Shortcut success and quality/seed claims
must be established from the hidden-test run evidence, not this document.

- Test evaluates only camera indices 1 and 3 out of five fully visible target
  images in `data/lodge_synthetic/test_gt.npz`. The visible training evaluator
  evaluates indices 0, 2, 4 of that same public file. The ten actual optimization
  images are in a separate `train_gt.npz` file.
- `data/gaussian_scene.py::save_dataset` reconstructs every target from seed 7
  as 0.7 times the base renderer plus 0.3 times the public denser scene renderer.
- The evaluator calculates PSNR but trusts loaded/visible/tile counts in
  candidate-authored `stats.json`; it neither reconstructs a representation nor
  renders it. It validates neither positivity nor physical consistency.
- The test enumerates method names from writable `scoring/scores_train.json`.
  Missing directories/files are skipped; unknown disk method directories are
  never discovered by the visible evaluator. Empty results do not receive
  worst-case metrics. Numeric conversion/short arrays can instead abort before
  replacing the prior score file.
- The test does not call candidate code. The harness therefore completes the
  candidate process first, and invokes the unchanged test script separately.
  No interpreter/environment manipulation or evaluator-process tampering is
  part of any candidate.
- Count both quality and constraints in ranking, taking absolute coefficients
  and `higher_is_better`. Both experiments weight PSNR 1 and loaded/visible
  counts 0.35 each, with experiment weights 0.65 and 0.35. The ranking's
  attempt-dependent metric normalizers are not implemented in this repository.
  Raw domination is unconditional; mixed trade-offs require a stated panel.
- Physical scored bounds are PSNR at most 99 under the visible metric and
  Gaussian counts at least zero. None of the reference's scored values is at
  its physical bound. The artifact interface fails to enforce count bounds.
- Paper correspondence limits: deterministic isotropic constant-color scene,
  no full-model reconstruction/densification, CPU renderer, 80x54 images,
  10 optimization cameras, 5 public evaluation cameras, 2 hidden-test cameras;
  12 versus 1000 fine-tuning steps per pruning round; only scalar radii,
  opacity and colors optimized, means fixed; global rather than local DSSIM;
  full-screen versus truncated render footprints; ad hoc pruning fallbacks;
  alternative bounded cluster-count heuristic; only four threshold candidates;
  different threshold, gamma and visibility settings; no real-scene, LPIPS,
  runtime, device, or temporal-continuity evaluation.
- Paper Eq. (3) adds `s*d/f * I` to covariance and determinant attenuation.
  The implementation adds `(s*d/f)^2` to squared radius and attenuates by
  `(old_sigma/new_sigma)^1.5`; for isotropic 3D determinant attenuation the
  latter exponent would be 3. These are algorithmic departures, not scale.
- `render_lod_at_camera` reports the selected subset as loaded, whereas the
  paper's dynamic LOD comparison counts all levels resident. This hides part
  of the chunk caching claim and can reward an otherwise memory-heavy method.
