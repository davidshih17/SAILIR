#!/bin/bash
# Medium-difficulty gravity3L benchmark: rank 1052/2101 (median) of the
# provided target list. t=8 r=8 s=2, FIRE answer = 1 master.
# Workers run the 2026-07-22 sector-raising fix (cone-exempt raw strip +
# one-step contract guard).
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1

TARGET="1,0,-2,1,1,1,1,1,1,1,0,0,0,0,0"
TAG=g1017
OUTDIR=$BASE/results/gr_reduce/$TAG
if [ -d "$OUTDIR" ]; then
    echo "REFUSING to overwrite existing $OUTDIR"; exit 1
fi
mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results
PYTHONUNBUFFERED=1 setsid $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
    --topology $BASE/topology_input/gravity3L --integral="$TARGET" \
    --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
    --model-checkpoint $BASE/checkpoints/gravity3L_canon10x_nosubs/best_model.pt \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --paper-masters-only --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
    --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
    --max-concurrent 100 --check-interval 5 --use-symmetry \
    > $OUTDIR/logs/orch.log 2>&1 < /dev/null &
echo "launched $TAG (PID=$!) -> $OUTDIR/logs/orch.log"
