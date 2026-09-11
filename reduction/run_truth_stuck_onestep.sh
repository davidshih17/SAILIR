#!/bin/bash
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
cd $BASE
export SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1
PYTHONUNBUFFERED=1 $PY reduction/truth_stuck_onestep.py > reduction/logs/truth_stuck_onestep_v3.log 2>&1
