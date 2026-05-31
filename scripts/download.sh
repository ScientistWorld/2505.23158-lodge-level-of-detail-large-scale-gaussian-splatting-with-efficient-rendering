#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: <1 MB
#
# This reproduction has no external downloads; the script only materializes
# deterministic workspace-local data for portability.

set -e

cd /home/user

# The compact reproduction uses a deterministic procedural Gaussian-splatting
# scene, so no network downloads are required.
python - <<'PY'
from pathlib import Path
from data.gaussian_scene import ensure_dataset
ensure_dataset(Path("data/lodge_synthetic"))
print("Prepared data/lodge_synthetic")
PY
