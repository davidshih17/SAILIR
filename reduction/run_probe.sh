#!/bin/bash
# Standalone one-step-worker PROBE on Condor: reduce ONE integral by ONE weight
# level (SAILIR_SUCCESS_TOTAL=1), as a single Condor job. This is the pre-reorg
# probe recipe (archive/probe_*.sh) adapted to the reduction/ layout
# (scripts/eval/beam_search_v7.py -> reduction/beam_search_v7.py). It is NOT the
# orchestrator (hierarchical_reduction.py) — no chaining to masters.
#
# Usage:
#   run_probe.sh <integral> <tag> [sym:0|1] [n_workers] [max_steps] [mem_gb]
# e.g.
#   run_probe.sh '0,1,1,1,1,1,1,1,-4,0,0' 74_off 0
#   run_probe.sh '0,1,1,1,1,1,1,1,-4,0,0' 74_on  1
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=$BASE/../RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=$BASE/topology_input/pentagonbox

INTEG="$1"; TAG="$2"; SYM="${3:-0}"; NW="${4:-8}"; MAXSTEPS="${5:-10000}"; MEM="${6:-48}"
D=$BASE/results/probe_${TAG}
rm -rf "$D"; mkdir -p "$D"

# Identical env to the success-only probe recipe; SAILIR_SYMMETRY=1 added for ON.
ENV="PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 SAILIR_SUCCESS_TOTAL=1 SAILIR_ACTION_SELECT=maxweight"
[ "$SYM" = "1" ] && ENV="$ENV SAILIR_SYMMETRY=1"
RCPUS=$((NW + 2))

cat > "$D/probe.sub" <<EOF
universe = vanilla
executable = $PY
arguments = -u $BASE/reduction/beam_search_v7.py --topology $TOPO --model $MODEL --integral='$INTEG' --output $D/result.pkl --ckpt $D/ckpt.pkl --ckpt-every 100 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 --max-steps $MAXSTEPS --max-actions 900 --beam-sort weight --no-paper-masters-only --prime 1009 --n-threads $NW --n-workers $NW --device cpu --model-batch-chunk 8
environment = "$ENV"
output = $D/probe.out
error  = $D/probe.err
log    = $D/probe.log
request_cpus = $RCPUS
request_memory = ${MEM}GB
request_disk = 20GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit "$D/probe.sub"
echo "probe $TAG (sym=$SYM, nw=$NW, max=$MAXSTEPS, mem=${MEM}GB) submitted -> $D"
echo "  time/steps: tail $D/probe.out          memory: grep 'Memory (MB)' $D/probe.log"
