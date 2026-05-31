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
  --out-dir checkpoints/lodge \
  --levels 2 \
  --gamma 0.060 \
  --visibility-threshold 0.030

bash scripts/evaluate.sh
