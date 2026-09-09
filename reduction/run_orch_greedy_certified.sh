#!/bin/bash
# Run the orchestrator with the CERTIFIED greedy worker.
#
# Not a variant of run_gr_hard_g1023_greedy.sh -- that script cannot be reused.
# It exports SAILIR_NM_PENALTY=0 and SAILIR_WORKER_V9=1, and the greedy dispatch
# aborts on both: nm_penalty is 0.1 in the certified configuration, and
# SAILIR_WORKER_GREEDY selects the worker now. Its --no-tabu and --beam-sort
# have no meaning either; tabu and beam sorting were stripped out of the module.
#
# WHAT THIS SCRIPT DOES NOT SET, ON PURPOSE
#   Every semantic variable comes from reduction/greedy_certified.json, pinned
#   into each Condor submit by hierarchical_reduction.greedy_env_string().
#   Setting them here would recreate exactly the drift the record removed. If
#   this shell contradicts the record, the orchestrator aborts before
#   submitting anything.
#
#   THE PRIME IS NOT OPTIONAL. hierarchical_reduction.py defaults --prime to
#   1009 while the p101 model was trained on a corpus generated entirely at
#   101; at 1009 the coefficient encoding is silently wrong. It is passed
#   explicitly below AND checked by greedy_check_certified().
#
# Usage:  run_orch_greedy_certified.sh <TARGET> <TAG>
#   TARGET  comma-separated integral, e.g. 1,1,1,1,1,1,1,1,2,2,0,0,0,0,0
#   TAG     output directory name under results/gr_reduce/
set -u
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

if [ $# -lt 2 ]; then
    echo "usage: $0 <TARGET> <TAG>" >&2
    exit 2
fi
TARGET="$1"
TAG="$2"

# Select the certified greedy worker. Everything else about the WORKER's
# configuration follows from the record, pinned into each Condor submit.
export SAILIR_WORKER_GREEDY=1

# ...but the ORCHESTRATOR PROCESS needs the topology in its OWN environment:
# hierarchical_reduction.main() imports topo_config, which refuses to guess a
# family. Leaving this out on the reasoning that "the record supplies it" gets
# a RuntimeError before anything is submitted -- the record supplies the
# workers, not this process. Taken from the record all the same, so the
# orchestrator and its workers cannot disagree about the family.
eval "$($PYTHON $B/scripts/greedy_env.py --export | grep SAILIR_TOPOLOGY)"
# Orchestrator-side only: routing and the delta substitution store. These
# govern how the orchestrator dispatches, not how a worker searches.
export SAILIR_ROUTE_CONDOR=1
export SAILIR_DELTA_SUBS=1
export PYTHONUNBUFFERED=1

# Read prime / cpus / checkpoint from the same record the workers use, so the
# command line cannot drift from the configuration it is supposed to run.
eval "$($PYTHON $B/scripts/greedy_env.py --cli)"

# Fail before submitting anything if the data files changed since certification
# -- a pinned path says WHICH file, not WHAT IS IN IT.
if ! $PYTHON $B/scripts/greedy_data_hashes.py --check; then
    echo "REFUSING: data inputs differ from the certified run (see above)." >&2
    exit 1
fi

OUTDIR=$B/results/gr_reduce/$TAG
LOG=$OUTDIR/logs/orch.log
if [ -e "$OUTDIR" ]; then
    echo "REFUSING: $OUTDIR exists (version the tag)" >&2
    exit 1
fi
mkdir -p $OUTDIR/logs $OUTDIR/work

echo "target     : $TARGET"
echo "prime      : $GREEDY_PRIME   (certified; the repo default 1009 is WRONG here)"
echo "v7-cpus    : $GREEDY_V7_CPUS"
echo "checkpoint : $GREEDY_CKPT"
echo "log        : $LOG"

setsid $PYTHON -u $B/reduction/hierarchical_reduction.py \
    --topology $B/topology_input/gravity3L --integral="$TARGET" \
    --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
    --model-checkpoint $B/$GREEDY_CKPT \
    --beam_width 1 --max_steps 1000000 --prime $GREEDY_PRIME \
    --paper-masters-only --use-v7-worker --v7-cpus $GREEDY_V7_CPUS \
    --worker-memory-gb 4 \
    --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
    --max-concurrent 1000 --check-interval 5 \
    > $LOG 2>&1 < /dev/null &

echo "launched pid $! -- NOT yet verified alive"
