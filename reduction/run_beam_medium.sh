#!/bin/bash
# Width-20 beam on a MEDIUM (96-step truth path) val reduction.
#   trajectory: data/shard_44/val.pt, 96 steps
#   integral  : 1,0,2,0,1,2,2,0,1,1,0,0,0,0,-1
#   sort      : prob_nm  (cost = lambda*nm - log(prob), lambda=0.03)
#   model     : bce / sigmoid checkpoint
# --max-steps 400 (~4x the truth length) bounds it rather than letting it run
# forever if it cannot close.
set -euo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
mkdir -p logs results
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""          # CPU; GPU 0 has the bce training
export SAILIR_TOPOLOGY=gravity3L
export SAILIR_BEAM_NM_LAMBDA=0.03

nohup python -u reduction/beam_search_v7.py \
    --topology    topology_input/gravity3L \
    --model       checkpoints/gravity3L_p101_bce/best_model.pt \
    --integral    1,0,2,0,1,2,2,0,1,1,0,0,0,0,-1 \
    --prime       101 \
    --beam-width  20 \
    --beam-sort   prob_nm \
    --max-steps   400 \
    --output      results/beam_medium96_probnm.pkl \
    > logs/beam_medium96_probnm.log 2>&1 &
echo $! > logs/beam_medium96_probnm.pid
echo "Started PID=$(cat logs/beam_medium96_probnm.pid)"
