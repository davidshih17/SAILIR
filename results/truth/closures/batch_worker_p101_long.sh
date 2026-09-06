#!/bin/bash
# p=101 retry worker. Two failure modes need DIFFERENT remedies:
#   capped   -> aborted at the 9 GB cap; give it 2 CPU / 20 GB (10 GB per CPU)
#   timeout  -> never approached the cap, ground to the 4h wall; give it 12h
# Args: <group> <outdir> <cap_mb> <wall_seconds>
set -u
export SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1 PYTHONUNBUFFERED=1
export SAILIR_KEEP_SYSTEMS=1 SAILIR_NO_SYSTEM_CACHE=1 SAILIR_PRIME=101
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
timeout "${4:-14400}" /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
  $B/results/truth/closures/batch_closure.py \
  --group "$1" --outdir "$2" --budget 500000 --cap-mb "${3:-9000}"
rc=$?
case $rc in
  42)  echo "[worker] exit=42 MEMORY CAP";;
  124) echo "[worker] exit=124 WALL-CLOCK TIMEOUT";;
  *)   echo "[worker] exit=$rc";;
esac
exit $rc
