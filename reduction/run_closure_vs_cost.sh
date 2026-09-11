#!/bin/bash
# Analysis only -- reads text logs, no GPU, no heavy memory. Login node is fine
# per the project rule (analysis of results is what the login node is for).
export SAILIR_TOPOLOGY=gravity3L
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHONUNBUFFERED=1 /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
  $B/reduction/closure_vs_cost.py \
  > $B/results/truth/finetune/closure_vs_cost.log 2>&1
