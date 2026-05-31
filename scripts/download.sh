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
# This reproduction has no external downloads. The deterministic synthetic
# scene is generated lazily by the baseline/method/eval scripts inside the
# configured container, where NumPy is installed by environment/setup.sh.

set -e

cd /home/user

mkdir -p data/downloads
echo "No external downloads required."
