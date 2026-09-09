#!/bin/bash
# Dispatch ONE target the way hierarchical_reduction.py really does it, and
# check the worker actually runs.
#
# check_greedy_dispatch.py verifies the submit file the orchestrator WRITES.
# That is not the same as verifying a job launched from it RUNS: the certified
# 125 go through greedy_worker.sh, which exports SAILIR_RAW_EQ_CACHE_CAP and
# SAILIR_KEEP_CKPT_EVERY and pins --v7-cpus 1. The orchestrator path supplies
# all of that differently, so "the orchestrator points at the greedy worker" is
# only true on paper until a job from it lands.
#
# This builds the submit file through create_condor_submit itself -- not a
# hand-written copy -- so what gets tested is the code path, not my idea of it.
set -u
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
D=$B/results/truth/finetune/orchdisp
PY=/het/p4/dshih/conda_envs/pyg4/bin/python

if [ -e "$D" ]; then echo "REFUSING: $D exists (version it)"; exit 1; fi
mkdir -p $D/logs $D/work/logs

# THE PRIME IS NOT OPTIONAL: this model was trained on a corpus generated at
# 101. Running it at the repo-wide default 1009 silently produces a wrong
# coefficient encoding -- no error, just wrong numbers.
export SAILIR_WORKER_GREEDY=1
export SAILIR_TOPOLOGY=gravity3L
PYTHONUNBUFFERED=1 $PY $B/scripts/make_orch_greedy_sub.py \
    --work-dir $D/work \
    --target 1,1,0,1,1,1,1,3,2,1,0,0,0,0,0 \
    --output $D/work/result.pkl \
    --topology $B/topology_input/gravity3L \
    --checkpoint $B/checkpoints/gravity3L_p101_scratch/best_model.pt \
    --prime 101 > $D/logs/make_sub.log 2>&1
rc=$?
echo "=== make_sub rc=$rc ==="
cat $D/logs/make_sub.log

if [ $rc -ne 0 ]; then echo "FAILED to build submit file"; exit 1; fi

SUB=$(grep -oP '^SUBMIT_FILE=\K.*' $D/logs/make_sub.log)
echo "=== submitting $SUB ==="
condor_submit "$SUB" 2>&1 | tail -2
