#!/bin/bash
# Hardening audit job: build the hidden world and public data, sweep the
# paper's free hyperparameters on public validation views, run the reference
# (3 seeds), better/weaker variants, simple rules and shortcut probes, score
# every case through the unchanged hidden test, then write the final scores.
set -e
cd /home/user
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python notes/hardening/audit.py
