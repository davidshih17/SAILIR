#!/bin/bash
# Analysis only: reads small text logs, no GPU, no heavy memory.
export SAILIR_TOPOLOGY=gravity3L
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHONUNBUFFERED=1 /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
  $B/reduction/compare_walk_lengths.py \
  > $B/results/truth/lex_corpus/walk_lengths.log 2>&1
