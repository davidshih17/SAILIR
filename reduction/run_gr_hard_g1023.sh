#!/bin/bash
# HARDEST gravity3L benchmark: rank 1/2101 of the provided target list.
# Full top sector (10 dens) + 2 dots, FIRE answer = 23 masters.
# Cache seeded from all 4 finished arms (7032 unique banked one-steps);
# workers run the 2026-07-22 sector-raising fix.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1

TARGET="1,1,1,1,1,1,1,1,2,2,0,0,0,0,0"
TAG=g1023
OUTDIR=$BASE/results/gr_reduce/$TAG
if [ -f "$OUTDIR/logs/orch_v22.log" ]; then
    echo "REFUSING: $OUTDIR/logs/orch_v22.log exists (version the log name)"; exit 1
fi
SAILIR_DELTA_SUBS=1 SAILIR_ROUTE_CONDOR=1 PYTHONUNBUFFERED=1 setsid $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
    --topology $BASE/topology_input/gravity3L --integral="$TARGET" \
    --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
    --model-checkpoint $BASE/checkpoints/gravity3L_canon10x_nosubs/best_model.pt \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --paper-masters-only --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
    --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
    --max-concurrent 1000 --check-interval 5 --use-symmetry --resume \
    > $OUTDIR/logs/orch_v22.log 2>&1 < /dev/null &
echo "launched $TAG (PID=$!) -> $OUTDIR/logs/orch_v22.log"
