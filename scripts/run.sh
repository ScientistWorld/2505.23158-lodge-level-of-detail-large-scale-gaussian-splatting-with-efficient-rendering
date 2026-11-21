#!/bin/bash
# Hardening: score every saved case through the final hidden test (final gates
# and coefficients), then write the final scores from the reference (seed 0).
set -e
cd /home/user
python notes/hardening/audit.py --stages cases,final
