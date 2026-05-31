#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

# Force this verification run to use the current source code, not stale
# generated artifacts from an earlier implementation.
rm -rf checkpoints/full_representation checkpoints/lod_dynamic checkpoints/nearest_chunk checkpoints/lodge
rm -rf data/lodge_synthetic
rm -f scoring/scores.json

bash scripts/download.sh
bash scripts/reproduce.sh
