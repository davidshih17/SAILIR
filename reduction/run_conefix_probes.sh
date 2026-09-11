#!/bin/bash
# PENTAGONBOX REGRESSION of the out-of-cone raw-strip fix (2026-07-22):
# identical recipe to run_primefix_probes.sh (the *_primefix references,
# 2026-07-14) re-run under the current code (topo_config general layer,
# generalized sector_rank/symmetry_route/canonical_masters/beam_search_v7,
# eval_coeff '/' branch). Gate: final master expressions identical to the
# references (cmp_conefix_probes.py).
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
export SAILIR_SECTOR_RANK=1

run_arm () {
    local TARGET="$1"; local TAG="$2"
    local OUTDIR=$BASE/results/ab_symmetry/$TAG/design1
    if [ -d "$OUTDIR" ]; then
        echo "  REFUSING to overwrite existing $OUTDIR"; return 1
    fi
    mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results
    PYTHONUNBUFFERED=1 setsid $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
        --topology $TOPOLOGY --integral="$TARGET" \
        --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
        --model-checkpoint $MODEL \
        --beam_width 40 --max_steps 1000000 --prime 1009 \
        --paper-masters-only --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
        --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
        --max-concurrent 100 --check-interval 5 --use-symmetry \
        > $OUTDIR/logs/orch.log 2>&1 < /dev/null &
    echo "  launched $TAG (PID=$!) -> $OUTDIR/logs/orch.log"
}

run_arm "1,-1,1,0,1,1,0,0,0,0,0"      gate_small_conefix
run_arm "2,0,1,0,1,-1,0,1,-1,-1,0"    m2_conefix
run_arm "1,1,-1,1,1,1,-1,1,-1,0,0"    m1_conefix
run_arm "1,1,0,1,1,0,-3,1,0,0,0"      m3_conefix
echo "4 regression arms launched."
