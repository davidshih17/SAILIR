#!/bin/bash
# Args: <recording.jsonl> <checkpoint> <outdir> [every]
export SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHONUNBUFFERED=1 exec /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
  $B/reduction/rank_on_recording.py \
  --recording "$1" --checkpoint "$2" --outdir "$3" --every "${4:-10}" --cpus 4
