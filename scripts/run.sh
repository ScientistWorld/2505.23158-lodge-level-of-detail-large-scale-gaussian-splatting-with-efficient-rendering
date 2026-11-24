#!/bin/bash
# Reproduce the reference end to end: check data, build the LODGE reference
# (seed 0) into checkpoints/lodge, then score the visible and hidden slices.
set -e
cd /home/user
export CUBLAS_WORKSPACE_CONFIG=:4096:8
bash scripts/download.sh
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
