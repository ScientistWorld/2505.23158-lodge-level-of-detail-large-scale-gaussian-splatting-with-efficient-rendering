# Hardening report (iteration 11)

**Short answer:** yes, better results now score higher, and every major shortcut in REDTEAM.md is closed. Reproducing that required rebuilding the gym rather than patching it.

The old test was 2 hidden views of an 848-Gaussian toy scene, with public, regenerable targets and method-reported costs. At that scale level-of-detail rendering (LOD, the paper's subject) did not even reduce work: 578 vs 574.5 visible Gaussians. Nothing in the old data could carry the paper's effect, so the data, evaluator, method and documents were all replaced. The old artefacts (and the red-team folder that probed them) are kept only as history in `redteam/`.

All evidence below is from the final hidden test (`bash scripts/evaluate_test.sh`, unchanged code, fresh process per case). Per-case score files and raw measurements are in `notes/hardening/cases/<case>/`. The audit harness is `notes/hardening/audit.py`; the rank analysis is `notes/hardening/analyze.py`, output in `notes/hardening/analysis.json`.

## What the gym is now

**World.** A synthetic street canyon, about 130 m long, built by `eval/test/world.py`, which lives only in `eval/test`:
- **True scene:** 558K dense anisotropic surface Gaussians with a procedural facade/road/object texture. It is hidden.
- **Targets:** 3×3-supersampled renders of the true scene. They are hidden for the test cameras; the training and validation targets are public.
- **Public base model G0:** 247,095 Gaussians at about 1.5× coarser spacing, trained for 5000 steps (L1 + D-SSIM) on the 209 training views. This is the paper's "full representation". Its validation PSNR went from 27.3 dB (untrained) to 41.1 dB.
- At 160×96 px and f = 96, G0 Gaussians are sub-pixel beyond about 10 m, so most of what a forward view sees is distant, redundant detail, the paper's regime.

**Submission.** `checkpoints/<method>/representation.npz`: a Gaussian pool, resident sets ("chunks") with anchor positions, optional per-Gaussian distance bands, and a nearest/two-nearest-blend flag. The format is in the `data/gsrender.py` docstring. The evaluator itself:
- selects the resident set per camera (Eq. 4 blending is part of the protocol);
- renders it with its own copy of the renderer (`eval/test/render.py`);
- counts loaded and visible Gaussians.

It reads nothing else.

**Experiments** (weights): `trajectory_views` 0.45, `flythrough` 0.20, `off_path_views` 0.35. See `scoring/EXPERIMENTS.md` and `briefing/evaluation.md`.

**Gates.** Failing any gate gives the worst value of every metric in every experiment.
- Invalid representation.
- Memory: peak resident Gaussians > 0.65·N0.
- Work: mean visible Gaussians > 0.70× the full model's, in any experiment.
- Quality: PSNR below the full model by more than 1.0 / 1.5 / 3.5 dB (trajectory / flythrough / off-path).
- Transitions: flythrough transition PSNR more than 0.5 dB below the full model's.

## 1. Do better results score higher?

**Rank definition.** The leaderboard normalisers depend on future attempts, so I use two stated panels (same rank formula as the red team):
- **H:** population standard deviation over the 10 honest runs that pass the gates.
- **A:** every case, failed runs entering with their worst values.

The rank is the weighted, coefficient-scaled, normalised improvement over `reference_seed0`.

**Before** (REDTEAM.md, old gym):
- The better variant (48/120 fine-tuning steps) gained +1.12 in panel H but −0.07 in panel B, inside B's 0.07 seed range.
- A 1000-step run lost quality but gained in B.
- The rank did not reliably reward improvement.

**After** (final hidden test, PSNR in dB, costs in Gaussians):

| Run | Trajectory PSNR / loaded / visible | Flythrough PSNR / transition | Off-path PSNR / loaded / visible | Rank H | Rank A |
|---|---|---|---|---:|---:|
| Full representation (fixed by the evaluator) | 41.181 / 247095 / 50817 | 37.498 / 33.245 | 33.567 / 247095 / 46366 | — | — |
| **Reference, seed 0** | 40.830 / 137678 / 31198 | 36.919 / 33.227 | 31.328 / 140104 / 29091 | 0 | 0 |
| Reference, seed 1 | 40.820 / 137812 / 31241 | 36.892 / 33.224 | 31.325 / 140263 / 29129 | −0.120 | −0.001 |
| Reference, seed 2 | 40.828 / 137732 / 31208 | 36.975 / 33.242 | 31.322 / 140180 / 29106 | +0.071 | +0.000 |
| Better: 3 LOD levels, seed 0 / 1 / 2 | 40.879 / 136950 / 30981 (seed 0) | 37.136 / 33.206 | 31.297 / 139394 / 29037 | +0.574 / +0.594 / +0.641 | +0.003 / +0.003 / +0.004 |
| Better: pseudo-view fine-tuning, seed 0 / 1 / 2 | 40.820 / 137639 / 31232 (seed 0) | 37.021 / 33.232 | 31.601 / 140068 / 29098 | +0.339 / +0.378 / +0.407 | +0.007 / +0.008 / +0.009 |
| Cheaper trade: γ = 0.06 (passes gates) | 40.656 / 114829 / 27244 | 36.476 / 33.157 | 30.455 / 116157 / 22926 | −0.941 | −0.010 |
| Weaker: no fine-tuning | 37.488 (fails quality) | 24.654 / 30.359 | 28.961 | worst | worst |
| Weaker: manual thresholds [40.5, 90] m | 40.884 / 161821 / 35205 (fails memory) | 37.078 / 33.242 | 32.625 / 164343 / 33907 | worst | worst |
| Weaker: γ = 0.10 | 39.873 (fails quality) | 35.702 | 29.898 | worst | worst |

How to read the table:
- **Seed noise is now negligible.** Across 3 reference seeds the spread is 0.010 dB (trajectory), 0.083 dB (flythrough) and 0.006 dB (off-path) in PSNR, and 134 loaded / 43 visible Gaussians. In rank units: 0.19 in panel H, 0.001 in A.
- **Both better variants beat the reference on every seed, in both panels, and by more than the noise.** Paired by seed, in panel H:
  - **3-level variant:** gains +0.57 to +0.71. It improves trajectory PSNR (+0.05 dB), flythrough PSNR (+0.22 dB) and every cost. Off-path PSNR (−0.03 dB) and transition PSNR (−0.02 dB) are essentially unchanged.
  - **Pseudo-view variant:** gains +0.34 to +0.50. It adds +0.27 to +0.36 dB off-path, roughly 45× the off-path seed spread. It distils from the full model at off-trajectory poses.
- **Weaker variants rank lower or fail.**
- **How seed noise was removed:** chunk construction is now deterministic (best-of-10 k-means with a fixed seed, fixed visibility-view seeds). Before that fix, the K-means seed moved chunk radii between 8 and 19 m and caused a 0.6 dB off-path seed spread (`notes/hardening/run1_seed_dependent_chunks/`).

## 2. Major shortcuts: fix, then score before and after

Before = REDTEAM.md (ΔB / ΔA panels of the old gym). After = final hidden test.

| Shortcut | Before | Fix | After |
|---|---|---|---|
| Copy the public target images (99 dB) | +4.54 / +4.69 | Submissions are Gaussian representations rendered by the evaluator, so images cannot be submitted. Test targets and test cameras exist only in `eval/test/hidden_set.npz`. | Not expressible |
| Regenerate the targets from the public generator and seed | +4.54 / +4.69 | The true scene and its generator live only in `eval/test/world.py`. Public data holds only the trained base model and public-view images. | Not expressible |
| Write zero or negative per-view costs | +2.78 / +4.05 | Costs are counted by the evaluator from the resident sets. `stats.json` and `method_info.json` are never read. | `sc_forged_files`: 0.000 (identical to the reference) |
| Constant black images / empty representation + zero costs | +2.08 / −0.73 | Quality floor gate. | `sc_empty_representation`: fails (7.78 dB), worst values |
| Black + impossible costs | +3.32 (A) | Costs computed by the evaluator; quality gate. | Same as above: fails |
| NaN pixels / non-finite parameters | +2.08 (B) | Validation rejects any non-finite value. | `sc_nan_means`: invalid, worst values |
| Replace the writable targets | +6.88 / +4.22 | The hidden test reads targets only from `eval/test`. | Not expressible |
| Edit the full-representation baseline artefacts | +4.89 / +5.13 | The baseline is rendered from the evaluator's own stored copy of G0. `checkpoints/full_representation` is ignored. | `sc_baseline_dir`: baseline unchanged, rank 0.000 |
| Delete every checkpoint (empty results, tie) | 0 / 0 | The `lodge` entry is required; missing means worst values. | `sc_missing_lodge`: worst values (H −669, A −3.22) |
| Empty training-score registry suppresses evaluation | 0 / 0 | The test discovers every `checkpoints/*` directory and never reads score files. | Not applicable |
| Unregistered extra method ignored | 0 / 0 | Every non-`_` directory is scored; invalid or failing ones get worst values. | `sc_extra_method`: extra entry scored as failed; `lodge` unchanged |
| Null costs / zero images crash and leave stale scores | stale file | Each method is wrapped; any error means worst values, and the scores file is always rewritten atomically. | `sc_bad_indices`, `sc_huge_pool`: worst values, fresh file |

Major shortcuts found in this iteration:

| Shortcut | Before the fix | Fix | After |
|---|---|---|---|
| **Submit the full model, or a less aggressive LOD, and win on quality.** Without a cost constraint, spread-normalisation lets small PSNR gains outweigh large memory and work increases. | Run 2, offline re-gating with memory ≤ N0 and no budget: `full_as_method` +1.034 in H; manual thresholds +0.575. Both above every better variant. | Device budget gates: peak resident ≤ 0.65·N0, mean visible ≤ 0.70× full. This is the paper's memory-constrained setting. | Both fail the memory budget and get worst values |
| **Drop opacity blending** (nearest chunk): 18% less memory, slightly higher trajectory PSNR, but popping | Passed all gates in run 2 | Transition gate: flythrough transition PSNR ≥ full − 0.5 dB | `ablation_nearest_chunk`: transition 32.106 vs 33.245, fails |
| **Exceed memory** by loading duplicates | Not probed | Memory gate | `sc_over_memory`: fails |
| **Oversized pool** | Not probed | Pool cap of 4·N0 | `sc_huge_pool`: invalid |
| **Hand-written LOD rules:** keep G0 within 12 m and a random 25% subsample with doubled scales beyond; or cull beyond 25 m | Not probed | None needed | `rule_distance_decimation` 30.06 dB, `rule_distance_cull` 30.36 dB: both fail the quality floor |

## 3. Shortcuts and weaknesses left open

- **Trading quality for cost inside the gates.** γ = 0.06 lowers PSNR by 0.17–0.87 dB and cuts resident memory by about 17%. It passes the gates and ranks below the reference in both stated panels (−0.94 H, −0.010 A). The sign of such trades still depends on the leaderboard normalisers. The memory budget leaves 4% above the reference (peak 0.623·N0); the work budget leaves 11–42%, depending on experiment and slice.
- **Training on the public validation views.** This inflates only the visible slice; the hidden cameras are different. It was not run.
- **Exhaustive threshold enumeration.** This is the brute force the greedy search avoids. With 10 candidates it costs 45 builds instead of 19, and it is an honest alternative rather than a loophole. Better thresholds are expressible and would be rewarded only through the measured metrics. Not run.
- **Distillation from the public base model at arbitrary poses** (the pseudo-view variant) is the main lever on `off_path_views`. It is allowed, as a legitimate improvement.
- **Fixed transition rules.** The protocol only offers nearest or two-nearest-blend transitions. Methods that want another rule must express it with more chunks or per-Gaussian distance bands.
- **Hidden-file trust.** `eval/test/hidden_set.npz` is tracked in the repository; the infrastructure must hide `eval/test/` from future agents.
- **Hardware.** The hidden test was run on a GPU. A CPU-only container runs the same code; I have not measured its runtime.

## 4. Reference score, before and after

- **Before** (old gym, cached hidden test, 2 views, 80×54 px, 848 Gaussians): PSNR 27.161, loaded 578.5, visible 550. Full: 27.036 / 848 / 574.5.
- **After** (`scoring/scores_test.json`, hidden test, reference seed 0):

| Experiment | LODGE PSNR | Full PSNR | Loaded | Visible | Other |
|---|---:|---:|---:|---:|---|
| `trajectory_views` | 40.830 | 41.181 | 137,678 (−44%) | 31,198 (−39%) | — |
| `flythrough` | 36.919 | 37.498 | — | — | transition 33.227 vs 33.245 |
| `off_path_views` | 31.328 | 33.567 | 140,104 (−43%) | 29,091 (−37%) | — |

- **Visible slice** (`scoring/scores_train.json`): trajectory 40.930 vs 41.055; flythrough 38.331 vs 38.693; off-path 30.925 vs 32.889. Budgets and gates are the same.
- **Scale:** the two worlds are not on the same scale. The paper's own numbers are kept, unchanged, in `scoring/reference.json`: the PSNRs as results and the counts under `_paper_results`.

### Paper claims reproduced on the hidden test

| Paper claim | This gym |
|---|---|
| **Core:** near-preserved quality with many fewer resident and visible Gaussians | −0.35 dB, −44% loaded, −39% visible (paper: −0.05 dB, −67%, −71%) |
| Dynamic LOD reduces work but keeps every level resident | 0.50× visible, 1.21× full memory |
| Chunks plus visibility filtering cut memory | 0.56× full |
| Opacity blending removes chunk-boundary popping at extra memory | nearest chunk: transition −1.12 dB, 18% less memory |
| Manual thresholds are worse than automatic ones | Partly. Manual [40.5, 90] m: +0.05 dB trajectory and +1.3 dB off-path, at +18% memory and +13% work, so it fails the budget. The paper reports manual thresholds worse on both quality and speed; here they are a costlier operating point, not dominated |
| Fine-tuning is needed | without it, −3.3 dB on trajectory views |

## 5. Headroom check

- **No scored metric sits at its bound.** PSNR sits below the full model (−0.35 / −0.58 / −2.24 dB) and far below its cap. Counts sit far above zero.
- **Added experiment: `off_path_views`** (weight 0.35). It targets the paper's stated robustness concern for cameras outside the training trajectory (supp. A.2). The reference loses 2.24 dB there. Pseudo-view distillation recovers 0.27–0.36 dB, so the headroom is real and reachable.
- **Simple baselines score clearly below the reference.** The full model fails the budget; the two hand-written rules lose 10–11 dB and fail.
- **No single simple variant reaches a bound.**

## 6. Weights and coefficients

The old experiments were replaced.
- **Old:** a compact quality/efficiency experiment (0.65) and an ablation over the method's own variants (0.35), where relabelling variants paid.
- **New weights:** 0.45 trajectory views (the paper's Tables 1 and 3 protocol), 0.20 flythrough (the opacity-blending claim) and 0.35 off-path (headroom).
- **PSNR coefficient 1.0, costs 0.35 each:** the paper leads with quality and then speed and memory.
- **`transition_psnr` reduced to 0.25** and flythrough PSNR made primary at 1.0. Every run that passes the gates sits within 0.04 dB of the full model on transitions, so at full weight the metric mostly added seed noise. Popping itself is enforced by the transition gate.

## 7. Departures from the paper corrected in the method

1. **Eq. 3 smoothing.**
   - Before: the radius was enlarged in quadrature by (s·d/f)², with opacity scaled by the ratio to the power 1.5, on isotropic Gaussians.
   - Now: the Mip-Splatting 3D filter, Σ + s·(d/f)²·I on anisotropic covariances, with the exact sqrt(det) opacity factor.
   - The printed "s·d/f" is dimensionally inconsistent, so I used the cited Mip-Splatting form.
2. **Fine-tuning.**
   - Before: 12 steps per round, colour/opacity/radius only, global DSSIM, and a training renderer that differed from the output renderer.
   - Now: the paper's 1000 steps per round; all parameters of the current level; windowed D-SSIM; the same renderer for training and evaluation.
3. **Chunk depth offset.** `d_l − radius` became `d_l + radius`. The paper's Table 3 text says the radius adjustment adds Gaussians through "finer LOD resolutions". The old sign produced 9–16 dB holes near chunk edges.
4. **Visibility filtering.** It now uses at least 32 random orientations per chunk. With about 6 views, up to 60% of each chunk was wrongly removed.
5. **Cluster count.** Eq. 5 is used unclipped (it was clipped to 2–8 before).
6. **Pruning and visibility fallbacks.** The quantile fallbacks were removed.
7. **Threshold candidates.** The greedy search now runs over 10 candidates instead of 4.
8. **Dynamic-LOD memory.** Its loaded count now includes every resident level.

## 8. What could not be fixed or is still scaled

- **Data.** The scene is synthetic, not Hierarchical-3DGS or Zip-NeRF data. Colours are RGB without spherical harmonics, the scene has 247K Gaussians, and the images are 160×96. The base model is trained without densification or opacity resets.
- **Hyperparameters.** The paper does not give γ, s or the visibility threshold for its LOD levels. γ = 0.03 and s = 0.2 were chosen on the public validation views (`notes/hardening/tuning.json`), before those views were made comparable to the test views.
- **Smaller effect than the paper's.** Reductions are −44% memory and −39% work, against the paper's −67% and −71%, at a slightly larger quality cost.
- **Metrics not reproduced.** FPS, mobile-device runs and LPIPS are not part of the gym. Costs are hardware-independent counts.
- **Unmeasured CPU runtime.** The hidden test's runtime on a CPU-only node is unknown; on one GPU it takes about 1 minute per method.
