#!/bin/bash
# Hardening: re-render comparable validation/test view sets (base model and
# training data unchanged), then score every saved case through the final
# hidden test (budget + transition gates) and write the final scores.
set -e
cd /home/user
python -m eval.test.world --rerender-eval-sets
python notes/hardening/audit.py --stages cases,final
