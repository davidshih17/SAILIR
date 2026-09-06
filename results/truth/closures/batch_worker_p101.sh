#!/bin/bash
# Same batched closure build, but in GF(101) instead of GF(1009).
# SAILIR_PRIME is read by truth_engine at import and propagated to ibp_env via
# TruthEngine.__init__, so the ENTIRE elimination runs at 101 -- pivots included.
# That matters: pivots are min(eq, key=tkey) over the surviving support, so a
# different prime can mask a different term and pick a DIFFERENT pivot. These
# trajectories are therefore not just 1009's with new coefficients.
set -u
export SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1 PYTHONUNBUFFERED=1
export SAILIR_KEEP_SYSTEMS=1 SAILIR_NO_SYSTEM_CACHE=1
export SAILIR_PRIME=101
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
timeout 14400 /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
  $B/results/truth/closures/batch_closure.py \
  --group "$1" --outdir "$2" --budget 500000 --cap-mb "${3:-9000}"
rc=$?
case $rc in
  42)  echo "[worker] exit=42 MEMORY CAP";;
  124) echo "[worker] exit=124 WALL-CLOCK TIMEOUT";;
  *)   echo "[worker] exit=$rc";;
esac
exit $rc
