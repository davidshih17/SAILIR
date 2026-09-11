#!/bin/bash
# Truth-engine reduction of the stuck g127 integral + FIRE-oracle gate.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE
export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1
mkdir -p results/truth
PYTHONUNBUFFERED=1 $PY reduction/truth_engine.py \
    --integral '1,1,1,1,1,1,1,0,0,-1,-1,0,0,0,0' \
    --output results/truth/g127_truth.pkl \
    --dr 1 --ds 1 \
    --trivialsector topology_input/gravity3L/kira_validate/sectormappings/GR/trivialsector \
    > reduction/logs/truth_g127_v3.log 2>&1
