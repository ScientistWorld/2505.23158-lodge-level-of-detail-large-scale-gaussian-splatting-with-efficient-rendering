#!/bin/bash
# Visible train-slice evaluator. Mirrors scripts/evaluate.sh's artifact
# contract: read method outputs from checkpoints/ and write a scores JSON.

set -e

cd /home/user

mkdir -p scoring

python eval/train/evaluate.py \
  --data-dir data/lodge_synthetic \
  --checkpoints-dir checkpoints \
  --reference scoring/reference.json \
  --scores scoring/scores_train.json
