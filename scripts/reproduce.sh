#!/bin/bash
# Reproduce the reference end to end: baseline template, LODGE, evaluation.
set -e
cd /home/user
bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --compare --post-design
