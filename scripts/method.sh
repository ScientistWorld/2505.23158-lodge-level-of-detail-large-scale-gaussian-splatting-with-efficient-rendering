#!/bin/bash
# Run the paper's method (LODGE) on the public data and write
# checkpoints/lodge/representation.npz (scored entry). Ablation variants of the
# same build (nearest chunk only, per-frame dynamic LOD) go to
# checkpoints/_ablation/, which evaluators do not score.
set -e
cd /home/user
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m method.lodge \
  --data-dir data/lodge_city \
  --out-dir checkpoints/lodge \
  --ablation-dir checkpoints/_ablation \
  --threshold-cache checkpoints/_cache/thresholds.json \
  --seed 0
bash scripts/evaluate.sh
