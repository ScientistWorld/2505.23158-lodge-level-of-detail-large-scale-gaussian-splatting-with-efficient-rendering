#!/bin/bash
# Hardening audit job (world and tuning already done): LODGE reference x3 seeds,
# better/weaker variants, trade-off probes, simple rules and shortcut probes
# through the unchanged hidden test, then final scores from the reference.
set -e
cd /home/user
python notes/hardening/audit.py
