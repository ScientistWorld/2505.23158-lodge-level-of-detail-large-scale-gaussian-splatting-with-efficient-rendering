#!/bin/bash
# Run the train/test scoring split against all existing checkpoint artifacts.

set -e

cd /home/user

rm -f scoring/scores_train.json scoring/scores_test.json

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
