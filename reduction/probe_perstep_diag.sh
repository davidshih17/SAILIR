#!/bin/bash
# DIAGNOSTIC PROBE — does per-step symmetry live-lock, or blow up compute/memory?
# Re-enables symmetry every step (SAILIR_SYMMETRY_PERSTEP=1) + instrumentation
# (SAILIR_SYM_INSTRUMENT=1) on the corner integral that hung earlier
# (0,1,1,0,1,0,0,1,0,0,0 = (4,0); model raises it to (5,1), then per-step
# symmetry hits the raised terms). Orchestrator-matching config (pentagonbox_nosym
# + --paper-masters-only + serial + SUCCESS_TOTAL=1), capped so it can't run away.
#
# Read the per-step log:
#   [sym-instr step N] repeat_states=...   <- climbs => LIVE-LOCK (returned states)
#   [v6 step N] ... t_step=.. rs_vsz=..     <- climbs => COMPUTE / MEMORY blowup
#
# Usage: probe_perstep_diag.sh [integral] [tag] [perstep 0|1] [max_steps]
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=$BASE/../RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=$BASE/topology_input/pentagonbox_nosym

INTEG="${1:-0,1,1,0,1,0,0,1,0,0,0}"
TAG="${2:-perstep_diag}"
PERSTEP="${3:-1}"          # 1 = per-step symmetry (TEST), 0 = step-0-only (BASELINE)
MAXSTEPS="${4:-60}"
D=$BASE/results/probe_${TAG}
rm -rf "$D"; mkdir -p "$D"

ENV="PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 SAILIR_SUCCESS_TOTAL=1 SAILIR_ACTION_SELECT=maxweight SAILIR_SYMMETRY=1 SAILIR_SYM_INSTRUMENT=1"
[ "$PERSTEP" = "1" ] && ENV="$ENV SAILIR_SYMMETRY_PERSTEP=1"

cat > "$D/probe.sub" <<EOF
universe = vanilla
executable = $PY
arguments = -u $BASE/reduction/beam_search_v7.py --topology $TOPO --model $MODEL --integral='$INTEG' --output $D/result.pkl --ckpt $D/ckpt.pkl --ckpt-every 100000 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 --max-steps $MAXSTEPS --max-actions 900 --beam-sort weight --paper-masters-only --prime 1009 --n-threads 1 --n-workers 1 --device cpu --model-batch-chunk 8
environment = "$ENV"
output = $D/probe.out
error  = $D/probe.err
log    = $D/probe.log
request_cpus = 2
request_memory = 16GB
request_disk = 20GB
Requirements = (TARGET.KFlops > 3000000)
priority = 1000000000
+JobFlavour = "workday"
queue
EOF

condor_submit "$D/probe.sub"
echo "perstep-diag '$INTEG' (perstep=$PERSTEP, max=$MAXSTEPS) -> $D"
echo "  watch: tail -f $D/probe.out"