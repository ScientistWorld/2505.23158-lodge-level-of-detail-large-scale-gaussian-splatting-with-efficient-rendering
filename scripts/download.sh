#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: 0 GB (nothing is fetched)
# Estimated disk usage: ~20 MB (committed under data/lodge_city)
#
# The benchmark scene is synthetic and agent-generated. Its public part
# (base Gaussian reconstruction, cameras, training/validation images) is small
# and committed to git under data/lodge_city/, so a fresh clone already has it.
# No external dataset or pretrained weights are used. This script only checks
# that the committed files are present.
set -e
cd /home/user
mkdir -p data/downloads
for f in data/lodge_city/base_gaussians.npz data/lodge_city/cameras.json data/lodge_city/images.npz; do
  if [ ! -f "$f" ]; then
    echo "missing $f: it is tracked in git; run 'git checkout -- data/lodge_city' to restore it" >&2
    exit 1
  fi
done
echo "All data present (committed in data/lodge_city)."
