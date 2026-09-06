#!/bin/bash
# Probe whether TruthEngine (and therefore on-demand closure regeneration)
# can run on this box, given topo_config.ROOT points at the other server.
set -uo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
mkdir -p logs
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" SAILIR_TOPOLOGY=gravity3L
nohup python -u scripts/probe_truth_engine_local.py > logs/probe_truth_engine.log 2>&1 &
echo "PID=$! -> logs/probe_truth_engine.log"
