#!/bin/bash
# HARDEST gravity3L benchmark (rank 1/2101), reduced with the p=101 model in
# GREEDY mode as the worker.
#
# Same orchestrator, same target, same symmetry/routing as run_gr_hard_g1023.sh.
# What changes is the WORKER: width-1 greedy instead of a 40-wide beam.
#
# WHY. On the 125-integral test set greedy solved 125/125 in 2.49 CPU-h against
# 160.7 for the best beam config (65x), and 300/300 on a held-out set. Every
# beam failure diagnosed on 1_0_1_0_0_0_5_1_1_1 -- which defeated a 40-wide beam
# for 34.5h and fell to greedy in 10.6s -- was a CROSS-STATE comparison
# discarding a good state. At width 1 no such comparison exists.
#
# THE WORKER MUST BE v9. beam_search_v7 contains no SAILIR_SCORE and no
# SAILIR_NM_PENALTY, so `--beam_width 1` under v7 is NOT the validated greedy
# config. SAILIR_WORKER_V9=1 selects onestep_worker_v9.py, whose CLI is
# identical (22 shared flags, 0 differences) and whose result.pkl is
# orchestrator-format, so reaping is unchanged.
#
# NM_PENALTY=0 is load-bearing: sort_prob is (action log-prob - penalty*nm), so
# a non-zero penalty perturbs the argmax and width 1 stops being greedy.
#
# NO --use-symmetry. The orchestrator hard-requires prime 1009 for symmetry
# (hierarchical_reduction.py:762) because symmetry_route/canonicalize transforms
# are built at 1009, and this model requires 101 -- the two are mutually
# exclusive. Running without symmetry means no orbit dedup, so more distinct
# integrals must be reduced; symmetry is production for gravity3L, so this run
# is strictly harder than the p=1009 symmetric baseline it will be compared to.
#
# PRIME 101, NOT 1009. This model was trained on a p=101 corpus; running it at
# 1009 gives a wrong coefficient encoding SILENTLY. That also means the banked
# one-steps under results/gr_reduce/g1023/work (prime 1009) are NOT reusable --
# hence a FRESH work dir and NO --resume.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1
# --- greedy worker configuration (forwarded into every Condor worker) ---
export SAILIR_WORKER_V9=1
export SAILIR_SCORE=local
export SAILIR_NM_PENALTY=0
export SAILIR_BEAM_TOTAL=1
export SAILIR_BEAM_SORT=prob
export SAILIR_WORKER_EXTRA_FLAGS='--no-tabu'

TARGET="1,1,1,1,1,1,1,1,2,2,0,0,0,0,0"
TAG=g1023_greedy
OUTDIR=$BASE/results/gr_reduce/$TAG
LOG=$OUTDIR/logs/orch_greedy_v2.log
if [ -f "$LOG" ]; then
    echo "REFUSING: $LOG exists (version the log name)"; exit 1
fi
mkdir -p $OUTDIR/logs $OUTDIR/work

SAILIR_DELTA_SUBS=1 SAILIR_ROUTE_CONDOR=1 PYTHONUNBUFFERED=1 setsid $PYTHON -u \
    $BASE/reduction/hierarchical_reduction.py \
    --topology $BASE/topology_input/gravity3L --integral="$TARGET" \
    --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
    --model-checkpoint $BASE/checkpoints/gravity3L_p101_scratch/best_model.pt \
    --beam_width 1 --beam-sort prob --max_steps 1000000 --prime 101 \
    --paper-masters-only --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
    --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
    --max-concurrent 1000 --check-interval 5 \
    > $LOG 2>&1 < /dev/null &
echo "launched $TAG (PID=$!) -> $LOG"
