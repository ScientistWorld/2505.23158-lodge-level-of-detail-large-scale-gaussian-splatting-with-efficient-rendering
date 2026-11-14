#!/bin/bash
# Re-run the actual reproduction pipeline, then refresh train/test split scores.

set -e

cd /home/user

rm -rf checkpoints/full_representation checkpoints/lod_dynamic checkpoints/nearest_chunk checkpoints/lodge
rm -rf data/lodge_synthetic
rm -f scoring/scores.json scoring/scores_train.json scoring/scores_test.json

bash scripts/download.sh
bash scripts/reproduce.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
