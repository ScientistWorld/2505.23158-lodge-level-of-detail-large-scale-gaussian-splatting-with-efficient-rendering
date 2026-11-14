#!/bin/bash
# Hidden test evaluator. Runs with no arguments and scores every method
# directory under checkpoints/ (plus the required `lodge` entry) by rendering
# its representation.npz at hidden cameras. Writes scoring/scores_test.json.
set -e
cd /home/user
mkdir -p scoring
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m eval.test.evaluate --checkpoints-dir checkpoints --scores scoring/scores_test.json
