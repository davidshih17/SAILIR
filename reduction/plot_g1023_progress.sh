#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
G=$B/results/gr_reduce/g1023
LOG="$G/logs/orch_v1.log $G/logs/orch_v2.log $G/logs/orch_v3.log $G/logs/orch_v4.log $G/logs/orch_v5.log $G/logs/orch_v6.log $G/logs/orch_v7.log $G/logs/orch_v8.log $G/logs/orch_v9.log $G/logs/orch_v10.log $G/logs/orch_v11.log $G/logs/orch_v12.log $G/logs/orch_v13.log $G/logs/orch_v14.log $G/logs/orch_v15.log $G/logs/orch_v16.log $G/logs/orch_v17.log $G/logs/orch_v18.log $G/logs/orch_v19.log $G/logs/orch_v20.log $G/logs/orch_v21.log $G/logs/orch_v22.log" \
  CSV=$G/nonmasters_vs_iter.csv PNG=$G/nonmasters_vs_iter.png \
  PYTHONUNBUFFERED=1 /het/p4/dshih/conda_envs/pyg4/bin/python \
  $B/archive/archive/_plot_nonmasters_vs_iter.py
