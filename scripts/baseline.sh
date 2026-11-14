#!/bin/bash
# Full-representation baseline. The evaluators compute this baseline
# themselves (entry `full_representation`), so nothing here is scored; this
# script writes the base model in the submission format as a template.
set -e
cd /home/user
python -m baseline.full_representation --data-dir data/lodge_city --out checkpoints/_examples/full_representation/representation.npz
