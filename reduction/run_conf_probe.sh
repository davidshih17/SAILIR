#!/bin/bash
# Off-manifold confidence probe: does the model stay maximally confident on
# states it has never seen? Runs a short beam (states after step ~1 are off any
# truth path) with SAILIR_CONF_PROBE=1, which logs the top score per task
# BEFORE beam selection (so the sort does not bias the sample).
# Compare against the ON-manifold baseline measured on val states.
set -euo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" SAILIR_TOPOLOGY=gravity3L
export SAILIR_BEAM_NM_LAMBDA=0.03 SAILIR_CONF_PROBE=1
CK=$1; TAG=$2
python -u reduction/beam_search_v7.py \
    --topology topology_input/gravity3L --model "$CK" \
    --integral 1,1,0,1,1,2,1,-1,1,1,0,0,-1,0,0 \
    --prime 101 --beam-width 20 --beam-sort prob_nm --max-steps 25 \
    --output /tmp/confprobe_$TAG.pkl > logs/confprobe_$TAG.log 2>&1 || true
echo "--- $TAG ---"; grep CONFPROBE logs/confprobe_$TAG.log | head -25
