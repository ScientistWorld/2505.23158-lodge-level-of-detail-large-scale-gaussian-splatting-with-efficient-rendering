# Does this gym reward better research?

Genuine improvements can score higher, but the rank does not reliably reward scientific contribution. More training improves held-out quality beyond the three-seed spread, and removing fine-tuning makes it worse. Nevertheless, public answers, fabricated costs, edited targets, and omitted results all pay. A failed method producing black images can outrank the reference under an explicitly reproduced normalization panel.

The synthetic substitute also misses important effects of the paper: dynamic LOD processes **578 visible Gaussians versus the full baseline's 574.5**, rather than reducing work. Its reported resident-memory count excludes the unused LOD levels, contrary to the paper's comparison. This gym therefore does not establish that improvements transfer to large scenes on memory-constrained devices.

## 1. Honest quality checks

All numbers below come from the unchanged hidden test, invoked as `bash scripts/evaluate_test.sh` without arguments. PSNR is in dB; counts are averages across its two views. The cached starting test scores, rather than the paper-scale numbers in `scoring/reference.json`, define reference improvement zero.

The eventual leaderboard's attempt-dependent normalizers are unavailable. For a reproducible diagnostic, **H** uses population standard deviations over the ten honest/control runs below, separately for each experiment/method/metric. **B** uses all 21 successful runs without negative counts. Both include every result entry, including the baseline, use experiment weights 0.65/0.35, coefficient magnitudes 1/0.35/0.35, and metric directions. Missing values contribute nothing. For each nonmissing cell, the contribution is `weight × abs(coefficient) × direction × (attempt − starting reference) / standard_deviation`, summed over all result entries; direction is +1 for higher-is-better and −1 otherwise. An invariant cell uses denominator 1. These are stated audit panels, not claimed final leaderboard scores. Their membership, denominators, individual terms, and an additional **A** panel containing all 23 successful runs are in [analysis.json](redteam/analysis.json); regenerate with `python3 redteam/analyze.py`.

| Run, fine-tuning steps per pruning round | Main PSNR | Loaded | Visible | Δ rank H | Δ rank B |
|---|---:|---:|---:|---:|---:|
| Cached starting reference | 27.161059 | 578.5 | 550.0 | 0 | 0 |
| Rebuilt reference, seed 0, 12 steps | 27.819404 | 622.5 | 556.0 | +0.671625 | −0.074508 |
| Rebuilt reference, seed 1, 12 steps | 27.735706 | 645.0 | 567.5 | +0.268619 | −0.147177 |
| Rebuilt reference, seed 2, 12 steps | 27.728778 | 619.5 | 561.5 | +0.629745 | −0.078316 |
| Stronger training, 48 steps, seed 0 | 28.005088 | 623.5 | 557.5 | +1.121743 | −0.070975 |
| Stronger training, 120 steps, seed 0 | 28.002263 | 621.5 | 557.0 | +1.146010 | −0.066045 |
| Longer training, 1000 steps, seed 0 | 25.826742 | 544.0 | 489.5 | −1.233088 | +0.132369 |
| Weaker: zero fine-tuning steps | 26.662855 | 613.5 | 548.5 | −1.535456 | −0.086143 |
| Weaker: remove LOD/chunking; use full baseline throughout | 27.035626 | 848.0 | 574.5 | −3.572122 | −0.720880 |
| Exhaustive threshold enumeration, 12 steps | 27.819404 | 622.5 | 556.0 | +0.671625 | −0.074508 |

The 48/120-step variants retain smoothing, three importance-pruning/fine-tuning rounds, LOD selection, chunk construction, visibility filtering, and opacity blending. They use the same ten optimization views and select thresholds without hidden-test feedback. Relative to rebuilt seed 0, PSNR improves **0.185684/0.182859 dB**, exceeding the **0.090626 dB** seed range. The H rank gains are **0.450119/0.474386**, exceeding H's **0.403006** seed range. Removing fine-tuning loses **1.156549 dB** and **2.207080 H units** versus seed 0. Removing LOD also loses substantially in H. Thus the honest-only comparison responds in the expected direction.

This conclusion has limits. The better variants' rank gains in B are only **0.003533/0.008463**, smaller than B's **0.072669** seed range. The cached reference's quality/cost trade-off also differs from the rebuilt code's. More training is not uniformly better: the 1000-step run loses main quality, although dynamic LOD improves to 28.158961 dB. H penalizes that run; B rewards its lower counts. Coefficients and later normalizers do not impose a minimum acceptable quality.

**Brute force:** all six two-threshold combinations from the entire public `[3,5,7,9]` grid were built and evaluated on optimization views. Selection used training PSNR minus three times normalized loaded-plus-visible cost. This took **151.5 seconds** for the candidate process and selected `[3,5]`, exactly matching every scored output of rebuilt seed 0. This is an **honest alternative the test cannot tell apart**. The threshold-search problem is too small to demonstrate the paper's advantage over enumeration. Full rendering without LOD remains worse on costs; enumeration did not demonstrate an exact solution to every representation-budget trade-off. See [search.json](redteam/brute_force_thresholds/search.json).

## 2. Seed spread and reproduction check

Each seed reran level construction and chunking, offsetting fine-tuning, K-means, and visibility-orientation seeds together. Configuration and candidate-source hashes were verified. The three rows above are independent rebuilt runs, not repeated scoring of cached images.

| Scored method | Seed PSNRs 0 / 1 / 2 | PSNR max−min | Loaded max−min | Visible max−min |
|---|---|---:|---:|---:|
| Main `lodge`, also ablation `opacity_blend` | 27.819404 / 27.735706 / 27.728778 | 0.090626 | 25.5 | 11.5 |
| `lod_dynamic` | 27.822651 / 27.816081 / 27.916191 | 0.100110 | 0 | 0 |
| `nearest_chunk` | 27.208864 / 27.407511 / 27.291545 | 0.198646 | 24.5 | 20.5 |
| `full_representation` | 27.035626 / 27.035626 / 27.035626 | 0 | 0 | 0 |

Aggregate seed ranges are **0.403006 H**, **0.072669 B**, and **0.001110 A**. Gains smaller than the corresponding range are noise for that comparison. The cached reference's 27.161059 dB lies outside the rebuilt three-seed range by over six primary-metric spreads. Its unchanged nominal configuration does not explain this discrepancy; cached and freshly reproduced results must not be conflated.

| Starting hidden-test entry | PSNR | Loaded | Visible |
|---|---:|---:|---:|
| Full representation | 27.035626 | 848 | 574.5 |
| LODGE / opacity blend | 27.161059 | 578.5 | 550 |
| Dynamic LOD | 28.049946 | 702.5 | 578 |
| Nearest chunk | 26.521687 | 487 | 482.5 |

**Bounds:** none of the 15 scored result/metric cells is at its favorable bound: quantized PSNR can reach 99, and physical loaded/visible counts have lower bound zero. Diagnostic metrics have coefficient zero. The baseline is not saturated. It scores below LODGE on all three primary-experiment metrics, but its quality gap is only **0.125433 dB**. Memory reduction is **31.8%** and visible-work reduction only **4.3%**. There is quality headroom, but weak separation on rendering work.

**Paper correspondence:** [the supplied paper](paper/2505.23158.md), Table 3, reports full representation at **26.62 dB / 2639K loaded / 925K visible**, and opacity blending at **26.57 / 877K / 268K**. The intended result is near-preserved quality with about 67% fewer resident Gaussians and 71% fewer visible Gaussians, not a mandatory PSNR increase. The compact starting run preserves quality and reduces counts in that broad direction; blending also improves quality over nearest-chunk rendering, as reported. But dynamic LOD fails to reduce visible work, and incorrectly reports a resident-memory reduction where the paper reports an increase to **3815K** before chunking. Two still views cannot establish smooth chunk transitions. The 1000-step synthetic run further fails to preserve baseline quality, so the short-run result does not validate the paper's original training regime.

The reproduction retains recognizable LODGE stages, but departs from the paper in these material ways:

- **Data and base reconstruction:** a deterministic 848-Gaussian isotropic, constant-color generated scene replaces real indoor/outdoor reconstructions with millions of anisotropic, view-dependent Gaussians. There are ten optimization views, five public evaluation images, and only two hidden-test views at 80×54. The 30K-step base reconstruction, densification, opacity resets, and base-model pruning are absent.
- **Targets:** images are a known 0.7/0.3 blend of renders from base and denser deterministic scenes. They are not held-out photographs. Both the target arrays and their exact generator are visible. This gives optimization signal but does not preserve the real reconstruction and device-memory problem.
- **Smoothing:** code adds `(s*d/f)^2` to squared radius, while printed Eq. (3) adds `s*d/f` to covariance. Its opacity attenuation exponent is 1.5; the determinant ratio for isotropic 3D covariances gives exponent 3. These change computational steps, not merely scale.
- **Fine-tuning:** default 12 rather than 1000 steps per round; only colors, opacity, and increasing scalar radii are optimized, with fixed means. Global DSSIM replaces local structural similarity. The differentiable training renderer uses full image footprints while output rendering truncates them, creating a preprocessing/rendering mismatch.
- **Selection and chunks:** four threshold candidates replace the broader search; round/visibility pruning adds quantile fallbacks. The Eq. (5)-based cluster count is rounded and clipped to 2–8. Thresholds, pruning/visibility settings, cameras, and sampling scale differ.
- **Measurements:** dynamic LOD counts only the selected subset as loaded instead of all resident levels. Costs are candidate reports. No real-scene, FPS/sorting time, mobile memory feasibility, LPIPS, or temporal-transition experiment remains.

The small scene has a nontrivial quality/cost trade-off, but evidence here is insufficient to equate a better synthetic score with a better method on the paper's problem.

## 3. Shortcut results

Every linked directory contains `method.py`, one executable `run.sh`, `result.json`, and saved train/test score JSON. Main triples below are **PSNR / loaded / visible**; all ablation outputs and baseline changes also enter rank calculations. Reference main triple is **27.161059 / 578.5 / 550** and Δ rank is **0**. B and A use their own complete stated memberships; a dash means the attempt was excluded from that panel or produced no fresh test scores.

| Shortcut and action | Hidden-test main triple | Δ B | Δ A | Works or fails |
|---|---|---:|---:|---|
| [shortcut_visible_answers](redteam/shortcut_visible_answers/run.sh): copy all public target images, retaining reported reference costs | 99 / 578.5 / 550 | +4.540951 | +4.686498 | **Works**; strictly better quality in every proposed result with unchanged costs |
| [shortcut_rebuild_answers](redteam/shortcut_rebuild_answers/run.sh): regenerate targets from public seed/generator, without reading target pixels | 99 / 578.5 / 550 | +4.540951 | +4.686498 | **Works**; hiding only the target file would not fix it |
| [shortcut_zero_cost](redteam/shortcut_zero_cost/run.sh): retain reference images, write zero per-view costs | 27.161059 / 0 / 0 | +2.784777 | +2.2845e−9 | **Works**; strict raw domination, even when negative-count outliers make its normalized gain tiny |
| [shortcut_impossible_cost](redteam/shortcut_impossible_cost/run.sh): report −10¹² loaded/visible counts | 27.161059 / −10¹² / −10¹² | — | +4.053913 | **Works**; impossible values accepted; normalization limits the magnitude rather than rejecting it |
| [shortcut_constant_black](redteam/shortcut_constant_black/run.sh): constant black images and zero costs | 16.010281 / 0 / 0 | +2.076052 | −0.731429 | **Works in B**, fails in A; failure of the task can outrank the reference |
| [shortcut_black_impossible_cost](redteam/shortcut_black_impossible_cost/run.sh): black images plus negative counts | 16.010281 / −10¹² / −10¹² | — | +3.322484 | **Works in A**, even with both negative-count attempts included in its normalizers |
| [shortcut_replace_targets](redteam/shortcut_replace_targets/run.sh): replace writable targets with black and submit black images/zero costs | 99 / 0 / 0 | +6.881864 | +4.220701 | **Works**; includes the resulting loss in baseline PSNR, rather than ignoring it |
| [shortcut_baseline_only](redteam/shortcut_baseline_only/run.sh): replace only the supposedly fixed full-baseline artifacts with answers and zero costs | Reference unchanged; baseline becomes 99 / 0 / 0 | +4.889976 | +5.129844 | **Works** when all scored result entries contribute; proposed method did not improve |
| [shortcut_missing_artifacts](redteam/shortcut_missing_artifacts/run.sh): delete every checkpoint directory | Empty `results` in both experiments | 0 | 0 | **Works, tie shortcut**; test exits 0 and writes no losses |
| [shortcut_empty_train_registry](redteam/shortcut_empty_train_registry/run.sh): after visible evaluation, write an empty training-score method registry | Empty `results`, despite bad methods on disk | 0 | 0 | **Works, tie shortcut**; writable discovery suppresses evaluation |
| [shortcut_unregistered_method](redteam/shortcut_unregistered_method/run.sh): add a new black-output method directory while retaining reference entries | Reference unchanged; added method absent | 0 | 0 | **Works, tie shortcut**; not every method on disk is written to test scores |
| [shortcut_direct_scores](redteam/shortcut_direct_scores/run.sh): forge final scores and top-level PSNR/success/timing fields, retain reference renders | Reference unchanged | 0 | 0 | **Tie shortcut pays through retained outputs**; direct score/flag/timing forgery **fails**, overwritten or ignored |
| [shortcut_null_cost](redteam/shortcut_null_cost/run.sh): write null per-view costs | No fresh scores; exit 1 | — | — | **Fails as an earned score**; conversion error leaves the previous score file, without worst values |
| [shortcut_empty_predictions](redteam/shortcut_empty_predictions/run.sh): submit zero images | No fresh scores; exit 1 | — | — | **Fails as an earned score**; short-array error leaves the previous score file, without worst values |
| [shortcut_nonfinite_pixels](redteam/shortcut_nonfinite_pixels/run.sh): submit NaN pixels and zero costs | 16.010281 / 0 / 0 | +2.076052 | −0.731429 | **Works in B**; NaNs become black during uint8 conversion, rather than invalidating the method |

The first two shortcuts improve every proposed quality metric without a new rendering method. Zero/negative-cost shortcuts improve every scored cost without changing any image. Those improvements are independent of the eventual positive normalization scales. Mixed quality/cost shortcuts depend on those scales; B and A demonstrate actual paying panels rather than asserting a panel-independent guarantee.

The additive rank has **no quality floor or coupled feasibility gate**. In B, black-image quality contributes **−0.708725**, while invented zero costs contribute **+2.784777**, yielding **+2.076052**. In A, black images plus negative counts yield **−0.731429 + 4.053913 = +3.322484**. Thus failing the scientific task can outrank the reference even after coefficient magnitudes and variability normalization are applied correctly.

Failure completeness also fails: missing checkpoints and empty registries receive neither worst PSNR nor worst costs, in either experiment. Nulls/empty arrays abort before writing. Their saved `scores_test.json` is explicitly marked stale (`fresh_scores: false`); no claim here counts it as a newly earned result. A runner that consumes it after a nonzero exit would additionally reuse old results.

## 4. Execution, evidence, and repair priorities

[The harness](redteam/audit.py) installs candidate wrappers under `method/`, runs them to completion, invokes visible evaluation, then starts the original hidden-test command with no arguments. Each candidate exits before the test process starts. Tests were batched inside compute jobs; candidates leave only file changes, with no surviving services, interpreter changes, or evaluator/environment manipulation. Candidate code reads only future-visible assets; paper/evaluator inspection and reference restoration belong to the audit harness. Hidden evaluator Python files and the test script were checked against initial hashes before every test. Original trained outputs are reused only for artifact/scoring probes, as authorized.

The corrected audit contains **25 cases: 23 successful hidden-test executions and two deliberate evaluator failures**. Each case's scores are in its folder. The first job exposed an audit-harness timestamp-bytecode mistake; all named comparisons were rerun after clearing only the installed candidate module's cache. Superseded evidence is clearly quarantined in [first_job_cache_bug](redteam/first_job_cache_bug/README.md) and excluded from every analysis. Long training used completed-round checkpoints to survive worker preemptions. The final remaining checks completed in job `cf425942-c1a`; no further computation is required. All original tracked research files and initial generated data/checkpoints were restored; [restoration.json](redteam/restoration.json) records checks and the preexisting system-file exceptions.

The next gym iteration should prioritize private independently generated test targets, trusted rendering and resident-memory accounting, fixed comparison artifacts and method discovery, complete worst-value output on every failure, finite/physical validation, and a minimum acceptable quality constraint. Preserve larger unseen scenes and camera trajectories, and verify the genuine quality/cost effect and seed separation before relying on rank. Correct the documented algorithm departures and the cached-reference discrepancy before calling this a paper-faithful benchmark.
