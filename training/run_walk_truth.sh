#!/bin/bash
set -euo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
mkdir -p logs
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
python -u training/walk_truth_reduction.py \
    --topology   topology_input/gravity3L \
    --shards_dir data \
    --n_scan_shards 40 \
    --checkpoint bce=checkpoints/gravity3L_p101_bce/best_model.pt \
    2>&1 | tee logs/walk_truth_reduction.log
