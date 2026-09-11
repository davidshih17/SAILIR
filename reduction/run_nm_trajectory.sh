#!/bin/bash
export SAILIR_TOPOLOGY=gravity3L
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHONUNBUFFERED=1 /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
  $B/reduction/nm_trajectory.py > $B/results/truth/finetune/hard4_ckpt/nm_traj.log 2>&1
