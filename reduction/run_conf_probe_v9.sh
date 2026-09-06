#!/bin/bash
# Off-manifold confidence probe on the PRODUCTION v9 pathway.
#
# v9 applies the upstream/fastmaxw cull -- the same action space the p=101
# corpus was generated in -- so these scores are directly comparable to the
# on-manifold scores measured on val states (which are in that space by
# construction, K=1000, verified: max 1000 actions/state, 33% at the cap).
#
# Logs the top score per (parent,target) task BEFORE beam selection, so the
# sort does not bias which states get sampled.
set -euo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
mkdir -p logs results
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" SAILIR_TOPOLOGY=gravity3L
export SAILIR_CONF_PROBE=1
CK=$1; TAG=$2
python -u reduction/beam_search_v9.py \
    --topology topology_input/gravity3L --model "$CK" \
    --integral 1,1,0,1,1,2,1,-1,1,1,0,0,-1,0,0 \
    --prime 101 --beam-width 20 --max-steps 30 \
    --output /tmp/confprobe_v9_$TAG.pkl > logs/confprobe_v9_$TAG.log 2>&1 || true
echo "--- $TAG (v9) ---"
grep -E "model_variant" logs/confprobe_v9_$TAG.log | head -1
grep CONFPROBE logs/confprobe_v9_$TAG.log | head -22
