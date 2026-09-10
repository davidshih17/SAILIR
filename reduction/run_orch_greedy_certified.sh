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
# SYMMETRY STORE at the model's prime. canonicalize_GR (hence symmetry_route,
# hence the orchestrator's Design-1 routing) reads these; without them it loads
# the 1009 store and the store-vs---prime guard refuses to start. The canonical
# SECTOR table needs no rebuild: rebuilding it from the 101 store reproduced
# results/canonical_sectors_GR_v2.pkl byte-identically, because sector
# canonicalization depends on WHICH maps exist, not on the field.
# The ORCHESTRATOR's own env, not just the workers'. routing_condor builds the
# routing worker.sh from os.environ, so anything missing here is silently
# defaulted into 7,144 jobs -- SAILIR_SECTOR_RANK absent meant it wrote =0 and
# every routing worker asserted on arrival.
export SAILIR_SECTOR_RANK=1
# THE STORE THAT IS ACTUALLY READ. topo_config.canonicalize_module() returns
# canonicalize2 for gravity3L -- NOT canonicalize_GR -- and canonicalize2 takes
# BOTH its transforms and its prime from this store's prod_point. Pointing here
# moves the whole symmetry layer (symmetry_route, sector_canon_maps,
# canonical_masters) to p=101 with one variable.
# results/gr_transforms*.pkl belongs to the GR-specific engine and is used only
# as gate_engine2's reference; setting it here did nothing for routing.
export SAILIR_SYM_STORE=results/gravity3L_transforms_v2_p101.pkl

# Discard any symmetry route producing more than this many terms and let a
# worker reduce the integral instead. Measured on the real frontier: s=20 gave
# a 1,315,600-term rule against a 714,538-term expression (8 masters in it),
# while IBP emits a median of 6-8 terms. The blowup is bimodal and NOT
# monotonic in s -- s=24 routed to 1 term, and the >=20 bucket was
# {1, 15, 28, 1315600} -- so the cap must be on the OUTPUT, not on s.
export SAILIR_ROUTE_MAX_TERMS=200
# Numerator-degree gate on routing. Checked BEFORE the expansion, so a skipped
# integral costs nothing -- MAX_TERMS only rejects after the terms are built.
# s is a proxy for predictability, not a cause; see greedy_certified.json.
export SAILIR_ROUTE_MAX_S=5
# The ONLY cap that acts DURING the expansion. MAX_TERMS rejects a rule after it
# is built and MAX_S screens only the top integral, so without this a single
# route can allocate millions of terms before anything stops it.
export SAILIR_ROUTE_MAX_EXPAND=20000
# ONE INTEGRAL PER CONDOR JOB. Routing cost is dominated by a small pathological
# tail; at 100/job a single slow integral blocked its 99 healthy neighbours AND
# withheld their results, because the worker writes its pkl only at the end --
# measured, 40 minutes in, 1,044 jobs had produced 2 outputs while ~16,000
# integrals were already routed but unwritten. Per-integral jobs make results
# incremental and the tail isolated. Startup does not argue against it: the
# worker needs only symmetry_route and a 200 KB store, measured at 0.02s.
export SAILIR_ROUTE_BATCH=1
# PER-INTEGRAL TIME LIMIT on symmetry routing. Measured over 104,383 integrals:
# median routes in 1.2s, p90 in 3.2min, but a few hundred ran for HOURS (one was
# still going at 6.7h), turning a pass that should be free into ~5,700 CPU-hours
# and defeating the purpose of routing. A timed-out integral becomes a survivor
# and is reduced by an IBP worker instead -- lossless, just less optimised.
export SAILIR_ROUTE_TIME_LIMIT=300
# SYMMETRY MELDED INTO THE WORKER, replacing the bulk pre-routing pass.
# greedy_worker._sym_first tries the route on its own target before loading
# torch: success returns steps=0 method='symmetry' in seconds, failure falls
# through to beam search at negligible cost. Same routing function as the bulk
# pass, but paid ON DEMAND by jobs that were going to run anyway, instead of
# speculatively over the whole frontier (~5,700 CPU-hours for one pass), and a
# slow route stalls one worker rather than the orchestrator.
export SAILIR_SYM_FIRST=1
export SAILIR_ROUTE_BULK=0
# Drip-feed: 104k jobs in one condor_submit jams the schedd (PITFALL 11).
export SAILIR_ROUTE_SUBMIT_CHUNK=5000
export SAILIR_ROUTE_QUEUE_CEILING=8000

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
RESUME=${RESUME:-0}

# LOG NAMING: never reuse a name. Each launch writes orch.log, orch_resume1.log,
# orch_resume2.log ... A resume that redirected to the existing orch.log would
# TRUNCATE it on open ('>'), destroying the record of the run being resumed --
# including the iteration history and every divergence decision.
LOG=$OUTDIR/logs/orch.log
if [ "$RESUME" = "1" ]; then
    n=1
    while [ -e "$OUTDIR/logs/orch_resume$n.log" ]; do n=$((n+1)); done
    LOG=$OUTDIR/logs/orch_resume$n.log
fi

if [ "$RESUME" = "1" ]; then
    [ -d "$OUTDIR/work/results" ] || { echo "REFUSING: no $OUTDIR/work/results to resume from" >&2; exit 1; }
    # COUNT WITHOUT A GLOB. `ls dir/*.pkl` expands to one argument per file and
    # blows ARG_MAX at this scale -- with 303,393 results it failed outright and
    # reported "RESUMING from 0", which reads exactly like a lost cache.
    _n_resume=$(find $OUTDIR/work/results -maxdepth 1 -name \*.pkl -printf . 2>/dev/null | wc -c)
    echo "RESUMING from $_n_resume completed worker results"
    RESUME_FLAG="--resume"
elif [ -e "$OUTDIR" ]; then
    echo "REFUSING: $OUTDIR exists (version the tag, or set RESUME=1)" >&2
    exit 1
else
    RESUME_FLAG=""
fi
mkdir -p $OUTDIR/logs $OUTDIR/work

# DIVERGENCE SWEEP. The memory cap cannot run on its own: the whole sweep sits
# behind `if args.diverge_nm > 0`, so --diverge-mem-mb alone is inert. Setting
# an unreachable nm threshold enables the block while leaving memory the only
# reason that ever fires (the kill triggers on EITHER reason, independently).
#
# This must be the orchestrator's OWN sweep, not an external watchdog: it does
# `del pending[integral]`, returning the slot. An outside condor_rm leaves the
# entry in `pending` forever and `available_slots = max_concurrent - len(pending)`
# decays to zero -- measured previously at 625 of 1000 slots dead after 7h.
# QUEUE DEPTH. --max-concurrent caps SUBMITTED jobs (running + idle), NOT
# running ones: hierarchical_reduction has no JobStatus==2 check anywhere, so
# how many actually run is Condor's fair share. At 1000 the queue drained
# faster than the orchestrator refilled it -- measured 430-614 running with
# idle=0 through most of a 56s median iteration (p90 131s, max 152s), i.e.
# slots sat empty waiting for the next submit while workers finish in ~6s.
# A deeper queue keeps idle jobs ready so Condor fills a freed slot at once.
# Stay well under the ~40k that once jammed the schedd; for larger batches see
# results/truth/p101_cull_corpus_v2/drip_submit.sh.
# --no-paper-masters-only, EXPLICITLY. The flag is BooleanOptionalAction with
# default=True, so merely OMITTING it leaves paper-masters-only ON -- a silent
# no-op. Measured: after removing the flag the frontier still reported L9r9:6
# rather than dropping the two corners to 4. The negation must be passed.
#
# The terminal set is paper masters UNION corner
# integrals in sectors the basis does not cover, which is what
# ibp_env.is_master already computes when PAPER_MASTERS_ONLY is False. The flag
# short-circuits that and makes corners non-terminal -- but IBP cannot reduce a
# corner within its own sector, and symmetry-first is off, so such a worker has
# no legal move and grinds until the memory guard kills it.
# 3000 was still not enough: idle fell back to 0 with running draining, because
# iterations grew to 142-208s as the frontier passed 680k, so the queue has to
# cover a ~3-minute gap between submits, not the ~50s it did at 600k.
# --use-symmetry (DESIGN 1 routing). OFF by default (action='store_true') and
# NOT used by any of the first four runs -- zero 'symmetry-routed' lines in
# orch.log, orch_resume1/2/3.log. That is the root cause of the stalled
# frontier, not the queue depth:
#   * the corpus sampled targets ONLY from the 298 canonical sectors (12,584 of
#     12,584), and every truth trajectory is pure IBP -- the truth engine never
#     calls canonical_monolithic_rule, it imports symmetry_route only for tkey.
#   * but IBP daughters land in SUBSECTORS, and a subsector of a canonical
#     sector need not be canonical. Measured: 213 of 307 sampled reductions from
#     a canonical parent (69%) emit at least one non-canonical daughter.
#   * a non-canonical integral's route down is a symmetry relabelling, which
#     IBP cannot perform -- the identity to the master ASCENDS the order. With
#     routing off the worker has no legal move and grinds to the memory guard.
#     Five of the six L9r9 frontier integrals map to a master in ONE term.
# Routing is free (cache entry, no worker), cascades to survivors, and is
# value-preserving, so it only ADDS cache entries -- the 281,815 already
# computed stay valid.
MAX_CONCURRENT=${MAX_CONCURRENT:-10000}

DIVERGE_MEM_MB=${DIVERGE_MEM_MB:-15000}
DIVERGE_FLAGS="--diverge-nm 100000000 --diverge-mem-mb $DIVERGE_MEM_MB"

echo "target     : $TARGET"
echo "prime      : $GREEDY_PRIME   (certified; the repo default 1009 is WRONG here)"
echo "v7-cpus    : $GREEDY_V7_CPUS"
echo "checkpoint : $GREEDY_CKPT"
echo "log        : $LOG"
echo "resume     : $RESUME   diverge: mem>${DIVERGE_MEM_MB}MB only"
echo "queue cap  : $MAX_CONCURRENT submitted (running = Condor fair-share, not capped here)"

setsid $PYTHON -u $B/reduction/hierarchical_reduction.py \
    --topology $B/topology_input/gravity3L --integral="$TARGET" \
    --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
    --model-checkpoint $B/$GREEDY_CKPT \
    --beam_width 1 --max_steps 1000000 --prime $GREEDY_PRIME \
    --no-paper-masters-only \
    --use-symmetry \
    --use-v7-worker --v7-cpus $GREEDY_V7_CPUS \
    --worker-memory-gb 4 \
    --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
    --max-concurrent $MAX_CONCURRENT --check-interval 5 \
    $RESUME_FLAG \
    $DIVERGE_FLAGS \
    > $LOG 2>&1 < /dev/null &

echo "launched pid $! -- NOT yet verified alive"
