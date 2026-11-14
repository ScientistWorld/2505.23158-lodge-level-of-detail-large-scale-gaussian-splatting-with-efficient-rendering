# Scoring audit artifacts

Read ../REDTEAM.md for conclusions. cases.json is the authoritative list of
25 valid cases. Each case has one run.sh, candidate method.py, result.json,
scores_train.json, scores_before_test.json, and scores_test.json. Check
result.json's test.fresh_scores before interpreting a saved score file.

Run a case with `bash redteam/<case>/run.sh` inside the gym compute container.
It installs and finishes the candidate, runs visible evaluation, then invokes
the original hidden test with no arguments. The harness restores the original
reference artifacts/data/scores afterwards. For empty_train_registry it also
installs the candidate's final empty registry after visible evaluation.

The initial finished-gym outputs are cached under ignored redteam/_cache/;
artifact-only probes reuse them. This cache and per-case binary checkpoints
remain in this workspace but are not committed. A fresh checkout needs the
original finished-gym artifacts to reproduce the cached-reference probes;
rebuilding today's method is measurably different, as the report explains.

Run `python3 redteam/analyze.py` to regenerate analysis.json using only saved
JSON and Python's standard library. It also verifies candidate wrapper hashes.
No model execution or packages are needed for this summary.

first_job_cache_bug/ contains invalid, superseded first-job evidence. It is
excluded from all case lists, panels, and reported conclusions. The audit
retains it to explain why the first nominal seed comparisons were rerun.

hidden_hashes.json records evaluator/test-script hashes checked before every
hidden-test invocation. restoration.json records final restoration checks.
The main report distinguishes independent domination from panel-dependent
trade-offs; no future leaderboard variance constants are assumed known.
