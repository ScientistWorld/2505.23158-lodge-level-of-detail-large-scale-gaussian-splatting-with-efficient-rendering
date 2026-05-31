#!/bin/bash
# Held-out test-slice evaluator. Mirrors scripts/evaluate.sh's artifact
# contract and, when available, scores the same method entries as train.

set -e

cd /home/user

mkdir -p scoring

python eval/test/evaluate.py \
  --data-dir data/lodge_synthetic \
  --checkpoints-dir checkpoints \
  --reference scoring/reference.json \
  --train-scores scoring/scores_train.json \
  --scores scoring/scores_test.json
