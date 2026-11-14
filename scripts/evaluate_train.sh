#!/bin/bash
# Visible evaluator on the public validation cameras (same protocol and gates
# as the hidden test). Scores every method directory under checkpoints/.
# Writes scoring/scores_train.json.
set -e
cd /home/user
mkdir -p scoring
python -m eval.train.evaluate --data-dir data/lodge_city --checkpoints-dir checkpoints --scores scoring/scores_train.json
