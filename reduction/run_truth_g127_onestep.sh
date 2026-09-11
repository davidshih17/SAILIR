#!/bin/bash
# Ground-truth ONE-STEP action sequence for the g127 target (worker semantics).
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE
export SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1
PYTHONUNBUFFERED=1 $PY reduction/truth_engine.py \
    --integral '1,1,1,1,1,1,1,0,0,-1,-1,0,0,0,0' \
    --output results/truth/g127_onestep_actions.pkl \
    --one-step-actions --dr 1 --ds 1 \
    --trivialsector topology_input/gravity3L/kira_validate/sectormappings/GR/trivialsector \
    > reduction/logs/truth_g127_onestep_v2.log 2>&1
