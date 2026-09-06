#!/bin/bash
# RE-VERIFY the 2-step drain on beam_search_v9 -- the DEPLOYED worker.
#
# Why v9 and not v7: v9 applies the upstream/fastmaxw ranked cull, which is the
# action space the p=101 corpus was generated in. v7 instead takes the FIRST
# --max-actions in enumeration order ("no notion of quality"), so it presents
# the model with an action space it never trained on. Every earlier v7 beam
# number in this session is confounded by that.
#
# Same trajectory as before: data/shard_44/val.pt, 96-step truth path.
set -euo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
mkdir -p logs results
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" SAILIR_TOPOLOGY=gravity3L
CK=${1:-checkpoints/gravity3L_p101_bce/best_model.pt}
TAG=${2:-v9verify}
python -u reduction/beam_search_v9.py \
    --topology topology_input/gravity3L --model "$CK" \
    --integral 1,0,2,0,1,2,2,0,1,1,0,0,0,0,-1 \
    --prime 101 --beam-width 20 --max-steps 400 \
    --output results/beam_${TAG}.pkl > logs/beam_${TAG}.log 2>&1 || true
echo "--- $TAG ---"
grep -E "model_variant|SUCCESS|FAIL|DONE|drained|best mw" logs/beam_${TAG}.log | tail -12
