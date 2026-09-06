#!/bin/bash
# Fine-tuning data generation, aligned with the deployed worker BY CONSTRUCTION:
# the sample's valid_actions IS the culled top-K list the worker hands the model
# at inference (upstream admissibility + fastmaxw ranking, K=1000, anchors
# uncapped), in rank order, with chosen_action_idx indexing into it.
#   $1 = integral (comma form)
set -u
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
D=$B/results/truth/truthcull
TAG=$(echo "$1" | tr ',' '_' | tr -d '-')
export SAILIR_TRUTHCULL_TRAIN=$D/ftdata1848/${TAG}.jsonl
METRIC=upstream exec $D/ab_worker.sh "$1" 1000 1000000
