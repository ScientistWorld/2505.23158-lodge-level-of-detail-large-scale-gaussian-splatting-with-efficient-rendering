#!/bin/bash
# Install Python packages into the container overlay (system Python, no venv).
# torch is the CUDA 12.1 build so GPU jobs can use the GPU; it also runs on CPU.
set -e
uv pip install --system numpy
uv pip install --system "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
