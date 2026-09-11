#!/bin/bash
# Direct one-step beam_search_v7 run with SAILIR_CANON=1 on a small symmetric
# integral, to confirm canonicalize-in-actions runs end-to-end (no crash) and the
# reduction only ever contains canonical integrals. Login-node sanity check, capped.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=$BASE/../RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=$BASE/topology_input/pentagonbox_nosym
INT="${1:-0,1,1,0,1,0,0,1,0,0,-1}"
MAX="${2:-60}"
D=$BASE/results/test_canon
rm -rf "$D"; mkdir -p "$D"
export PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 \
       SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 SAILIR_SUCCESS_TOTAL=1 \
       SAILIR_ACTION_SELECT=maxweight SAILIR_CANON=1
$PY -u $BASE/reduction/beam_search_v7.py --topology $TOPO --model $MODEL \
    --integral="$INT" --output $D/result.pkl --ckpt $D/ckpt.pkl --ckpt-every 100000 \
    --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 --max-steps $MAX \
    --max-actions 900 --beam-sort weight --paper-masters-only --prime 1009 \
    --n-threads 1 --n-workers 1 --device cpu --model-batch-chunk 8 \
    > $D/out.log 2>&1
echo "exit=$? -> $D/out.log"
