#!/bin/bash
# Pre-flight check on the p=101 gravity3L corpus before the 200-epoch run.
set -euo pipefail
cd /home/shih/work/SAILIR_p101

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn

mkdir -p logs
export PYTHONUNBUFFERED=1

python -u training/verify_corpus.py \
    --shards_dir data \
    --topology   topology_input/gravity3L \
    --prime      101 \
    --n_shards   20 \
    2>&1 | tee logs/verify_corpus.log
