#!/bin/bash
# g1023 ROUND 2 — resumes round 1's cache READ-ONLY, with the divergence sweep
# built into the orchestrator.
#
# WHY ROUND 1 STALLED (not a physics problem, a bookkeeping one): the external
# watchdog could only condor_rm a diverging worker; its entry stayed in
# `pending` forever because the reaper only clears an entry when an output file
# appears. `available_slots = max_concurrent - len(pending)` therefore decayed
# to zero. Measured at kill time: 625 of 1000 slots were corpses, only 210 real
# jobs on the cluster, and the orchestrator had stopped submitting.
#
# THE FIX (--diverge-nm): the sweep now runs INSIDE the orchestrator and does
# `del pending[integral]`, returning the slot. The integral is recorded to
# work/diverged_killed.jsonl and added to a `diverged` set that is subtracted
# from to_submit, so the freed slot is NOT immediately refilled by the same
# divergence -- that would be the kill/resubmit loop.
#
# --diverged-from SEEDS THE DIVERGED SET from round 1's kills. Those integrals
# produced no result, so they are NOT in the resumed cache and would all be
# re-dispatched: measured on the first round-2 attempt, 254 of 454 workers
# (56%) were re-running integrals round 1 had already killed as divergent.
#
# NOTHING FROM ROUND 1 IS TOUCHED. --resume-from loads the prior cache
# READ-ONLY ("the prior dir is never modified"); all new results, logs and
# submits go to the round-2 dir. Round 1 keeps its 95,852 result pkls,
# 625-entry killed_diverging.jsonl, orchestrator log and worker logs.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1
export SAILIR_WORKER_V9=1
export SAILIR_SCORE=local
export SAILIR_NM_PENALTY=0.1
export SAILIR_BEAM_TOTAL=1
export SAILIR_BEAM_SORT=prob
export SAILIR_WORKER_EXTRA_FLAGS='--no-tabu'

TARGET="1,1,1,1,1,1,1,1,2,2,0,0,0,0,0"
PREV=$BASE/results/gr_reduce/g1023_greedy_nm01
OUTDIR=$BASE/results/gr_reduce/g1023_greedy_nm01_r2
LOG=$OUTDIR/logs/orch_r2_v1.log
if [ -e "$LOG" ]; then echo "REFUSING: $LOG exists (version the log name)"; exit 1; fi
if [ ! -d "$PREV/work/results" ]; then echo "REFUSING: no round-1 cache at $PREV/work/results"; exit 1; fi
mkdir -p $OUTDIR/logs $OUTDIR/work

SAILIR_DELTA_SUBS=1 SAILIR_ROUTE_CONDOR=1 PYTHONUNBUFFERED=1 setsid $PYTHON -u \
    $BASE/reduction/hierarchical_reduction.py \
    --topology $BASE/topology_input/gravity3L --integral="$TARGET" \
    --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
    --resume-from $PREV/work \
    --diverged-from $PREV/killed_diverging.jsonl \
    --model-checkpoint $BASE/checkpoints/gravity3L_p101_scratch/best_model.pt \
    --beam_width 1 --beam-sort prob --max_steps 1000000 --prime 101 \
    --paper-masters-only --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
    --diverge-nm 500 --diverge-mem-mb 20480 \
    --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
    --max-concurrent 1000 --check-interval 5 \
    > $LOG 2>&1 < /dev/null &
echo "launched g1023 round 2 (PID=$!) -> $LOG"
