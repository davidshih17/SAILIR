#!/bin/bash
# G885 VALIDATION RUN of the symmetry-first architecture (2026-07-31).
#   - workers try the symmetry route before beam search (SAILIR_SYM_FIRST=1)
#   - orchestrator loop NEVER solves routes (SAILIR_ROUTE_LOOKUP_ONLY=1);
#     rules arrive as ordinary worker results
# Fresh work dir; the old banked g885 (results/gr_reduce/g885) is the
# reference for the final-state gate. max-concurrent capped at 300 so v22
# keeps its share of the cluster.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1

TARGET="1,0,1,-1,1,1,1,0,1,1,-1,0,0,0,0"
TAG=g885_symfirst
OUTDIR=$BASE/results/gr_reduce/$TAG
mkdir -p $OUTDIR/logs
if [ -f "$OUTDIR/logs/orch_v3.log" ]; then
    echo "REFUSING: $OUTDIR/logs/orch_v3.log exists (version the log name)"; exit 1
fi
SAILIR_DELTA_SUBS=1 SAILIR_SYM_FIRST=1 SAILIR_ROUTE_LOOKUP_ONLY=1 \
PYTHONUNBUFFERED=1 setsid $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
    --topology $BASE/topology_input/gravity3L --integral="$TARGET" \
    --output $OUTDIR/reduction.pkl --work-dir $OUTDIR/work \
    --model-checkpoint $BASE/checkpoints/gravity3L_canon10x_nosubs/best_model.pt \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --paper-masters-only --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
    --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
    --max-concurrent 300 --check-interval 5 --use-symmetry \
    > $OUTDIR/logs/orch_v3.log 2>&1 < /dev/null &
echo "launched $TAG (PID=$!) -> $OUTDIR/logs/orch_v3.log"
