#!/bin/bash
# Beam search on the SAME 844-step reduction the step-by-step walk used, with:
#   --beam-width 20      (not the default 40)
#   --beam-sort prob     rank ONLY by the current step's model probability --
#                        no n_non_masters progress regulariser, no accumulated
#                        multi-step log-prob sum. Added for this experiment;
#                        'weight'/'mixed' are untouched.
# Model: the bce/sigmoid checkpoint.
set -euo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
mkdir -p logs results
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""      # CPU: GPU 0 is running the bce training
export SAILIR_TOPOLOGY=gravity3L    # keys topo_config; default pentagonbox would assert

nohup python -u reduction/beam_search_v7.py \
    --topology    topology_input/gravity3L \
    --model       checkpoints/gravity3L_p101_bce/best_model.pt \
    --integral    1,2,1,1,1,1,1,0,-2,0,0,0,0,0,0 \
    --prime       101 \
    --beam-width  20 \
    --beam-sort   prob \
    --max-steps   2000 \
    --output      results/beam_prob20_bce.pkl \
    > logs/beam_prob20_bce.log 2>&1 &
echo $! > logs/beam_prob20_bce.pid
echo "Started PID=$(cat logs/beam_prob20_bce.pid). Tail: tail -f logs/beam_prob20_bce.log"
