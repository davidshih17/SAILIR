#!/bin/bash
# Post-strip regression: run ONLY the --use-symmetry (design1) arm for the four
# targets, fully detached. NO baseline arm (baselines already ran and are stored).
# Compare masters + worker counts against the stored pre-strip runs afterward.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox

run_sym () {
    local TARGET="$1"; local TAG="$2"
    local OUTDIR=$BASE/results/ab_symmetry/$TAG/design1
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
    echo "  launched design1 $TAG (PID=$!) -> $OUTDIR/logs/orch.log"
}

run_sym "1,-1,1,0,1,1,0,0,0,0,0"      gate_small_symcheck
run_sym "1,1,-1,1,1,1,-1,1,-1,0,0"    m1_symcheck
run_sym "2,0,1,0,1,-1,0,1,-1,-1,0"    m2_symcheck
run_sym "1,1,0,1,1,0,-3,1,0,0,0"      m3_symcheck
echo "4 design1 (--use-symmetry) arms launched."
