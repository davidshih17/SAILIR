#!/bin/bash
# Standalone probe: can SAILIR_SUCCESS_TOTAL=1 (reduce-by-total-ordering
# success) crack the g127 first level that the bucket-drain criterion is stuck
# on (plateau nm=27 since step ~350)? Single local worker, same model/settings.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE
export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1
export SAILIR_SUCCESS_TOTAL=1
mkdir -p results/gr_reduce/g127_st_probe
PYTHONUNBUFFERED=1 $PY reduction/onestep_worker_v7.py \
    --topology topology_input/gravity3L \
    --integral '1,1,1,1,1,1,1,0,0,-1,-1,0,0,0,0' \
    --output results/gr_reduce/g127_st_probe/probe.pkl \
    --model-checkpoint checkpoints/gravity3L_canon10x_nosubs/best_model.pt \
    --beam_width 40 --max_steps 2000 --prime 1009 \
    > reduction/logs/g127_st_probe_v1.log 2>&1
