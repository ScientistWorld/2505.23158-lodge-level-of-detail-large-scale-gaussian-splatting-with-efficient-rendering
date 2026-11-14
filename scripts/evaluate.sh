#!/bin/bash
# Standard evaluation entry point (public validation views).
#
# Uses only eval/ and data/ (never method/). Every method directory under
# checkpoints/ must contain representation.npz in the format documented in
# data/gsrender.py; the evaluator renders it itself and recomputes images,
# resident (loaded) Gaussians and visible Gaussians. Writes scoring/scores.json.
set -e
cd /home/user
mkdir -p scoring
python -m eval.train.evaluate --data-dir data/lodge_city --checkpoints-dir checkpoints --scores scoring/scores.json
