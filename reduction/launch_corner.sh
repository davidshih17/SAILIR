#!/bin/bash
# Reduce one low corner to masters via the orchestrator (Condor). Args: INT_STR IDX
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
INT="$1"; IDX="$2"
OUTDIR=$BASE/results/corner_reductions/c$IDX
mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results
PYTHONUNBUFFERED=1 $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
    --topology $TOPOLOGY --integral="$INT" \
    --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --no-paper-masters-only --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
    --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
    --max-concurrent 100 --check-interval 5 \
    > $OUTDIR/logs/orch.log 2>&1 &
echo "launched corner $IDX int=$INT PID=$! -> $OUTDIR/logs/orch.log"
