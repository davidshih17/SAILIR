#!/bin/bash
# Fully GREEDY solve (beam width 1) with the current BCE model.
#
#   $1  integral (comma-separated)   $2  tag   $3  max-steps
#
# beam-width 1 makes K = max(1, beam_width//2) = 1, so the search expands
# exactly the top-scoring action at each state and never backtracks -- greedy
# in both senses (one state kept, one child generated).
#
# Everything else matches the production v9 pathway: --prime 101 and
# --max-actions 1000, the K=1000 ranked cull the p=101 corpus was generated in.
# CPU-only so it does not contend with training on GPU 0.
set -uo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
mkdir -p logs results
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" SAILIR_TOPOLOGY=gravity3L

INT=${1:-1,0,1,0,0,0,5,1,1,1,0,0,0,0,0}
TAG=${2:-greedy}
STEPS=${3:-400}
CK=${4:-checkpoints/gravity3L_p101_bce/best_model.pt}

echo "integral=$INT  model=$CK  beam-width=1  max-steps=$STEPS"
python -u reduction/beam_search_v9.py \
    --topology topology_input/gravity3L --model "$CK" \
    --integral "$INT" \
    --prime 101 --beam-width 1 --max-steps "$STEPS" \
    --max-actions 1000 \
    --output results/greedy_${TAG}.pkl > logs/greedy_${TAG}.log 2>&1
rc=$?
echo "exit=$rc"
grep -E "SUCCESS|FAIL|DONE|drained|best state|path_len|peak_rss" logs/greedy_${TAG}.log | tail -8
