#!/bin/bash
# Hardening audit job: renderer timing, then the audit harness (world already
# built: tune gamma/s on public validation views, LODGE reference x3 seeds,
# variants, simple rules, shortcut probes through the unchanged hidden test,
# final scores).
set -e
cd /home/user
python notes/hardening/profile_render.py 2>&1 | head -14
python notes/hardening/audit.py
