#!/bin/bash
# Hardening smoke test: tiny world, every stage of the audit harness.
set -e
cd /home/user
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi || true
nproc
python -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__)"
rm -rf tmp/audit_smoke
python notes/hardening/audit.py --smoke
