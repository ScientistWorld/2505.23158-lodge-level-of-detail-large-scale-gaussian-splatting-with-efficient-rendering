#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python -m method.lodge \
  --data-dir data/lodge_synthetic \
  --out-dir checkpoints/lod_dynamic \
  --levels 2 \
  --gamma 0.055 \
  --visibility-threshold 0.220 \
  --smooth-s 1.0 \
  --variant lod_dynamic

python -m method.lodge \
  --data-dir data/lodge_synthetic \
  --out-dir checkpoints/nearest_chunk \
  --levels 2 \
  --gamma 0.055 \
  --visibility-threshold 0.220 \
  --smooth-s 1.0 \
  --variant nearest_chunk

python -m method.lodge \
  --data-dir data/lodge_synthetic \
  --out-dir checkpoints/lodge \
  --levels 2 \
  --gamma 0.055 \
  --visibility-threshold 0.220 \
  --smooth-s 1.0 \
  --variant opacity_blend

bash scripts/evaluate.sh
