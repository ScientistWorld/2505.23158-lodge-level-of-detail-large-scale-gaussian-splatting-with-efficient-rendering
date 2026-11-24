#!/bin/bash
# CPU-only check of the hidden test: runtime and agreement with the stored GPU scores.
set -e
cd /home/user
mkdir -p tmp/cpu_check
python -c "import torch; print('cuda available:', torch.cuda.is_available(), 'threads:', torch.get_num_threads())"
start=$(date +%s)
python -m eval.test.evaluate --checkpoints-dir checkpoints --scores tmp/cpu_check/scores_test.json
end=$(date +%s)
echo "CPU hidden test seconds: $((end-start))" | tee notes/hardening/cpu_check_time.txt
python notes/hardening/cpu_check.py tmp/cpu_check/scores_test.json
