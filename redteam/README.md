# Scoring audit

Run `bash redteam/run_all.sh` **as a compute job**, or run an individual
`redteam/<case>/run.sh` in the gym container. Each case installs its own
`method.py` as `method/audit_candidate.py`, runs it to completion, runs the
visible evaluator, then starts the unmodified hidden test in a separate
process with `bash scripts/evaluate_test.sh` and no arguments. No candidate
process survives into evaluation. No interpreter packages, import paths,
environment controls, evaluator code, or test scripts are modified.

The harness backs up the original generated data and checkpoint outputs once
in ignored `_cache/`, restores them before and after each case, and checks
hashes of all evaluator Python sources and the test script before testing.
Only the harness reads the hidden-test scores; candidates use the visible
data, method, baseline, checkpoint interface, and training images.

The cached reference outputs are reused for artifact-only shortcuts as
explicitly permitted by the audit. Three reference seeds and the longer
training and ablation cases actually rebuild the representation. Seed offsets
0, 1, 2 change training threshold jitter, chunk initialization, and orientation
augmentation; the synthetic scene and camera split stay unchanged.

Each case keeps `scores_train.json`, `scores_test.json`, `result.json`, and
the pre-test score file. A failed test's copied `scores_test.json` is marked
`fresh_scores: false` in `result.json`: it is a stale file, **not a score earned
by that candidate**. Generated binary checkpoint archives and detailed stdout
logs are ignored; the small score and status evidence is tracked.

`shortcut_*` folders are explicitly scoring shortcuts, including tied entries
that hide failed methods. The direct-score, null-cost and empty-prediction
folders include probes that may fail; their status must be checked rather
than interpreting a leftover score file as success. Quality variants and
exhaustive search are honest algorithmic comparisons.

The ranking normalizers depend on future attempts and are not available at
audit time. Report raw scored metrics and dominance first. Any normalized
aggregate computed from an audit comparison panel is explicitly labeled as
panel-dependent, not the unavailable final leaderboard rank.
