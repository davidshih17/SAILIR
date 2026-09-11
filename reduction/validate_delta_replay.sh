#!/bin/bash
# DELTA-LOOP REPLAY VALIDATION: for each COMPLETED arm, re-run the orchestrator
# under the incremental-substitution code with --resume on a hard-linked COPY of
# the arm's work dir. The cache is complete, so the whole reduction happens by
# pure substitution (no workers); the final expression must be IDENTICAL to the
# arm's banked reduction.pkl (produced by the old full-fixpoint loop).
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1

declare -A TGT
TGT[g885]="1,0,1,-1,1,1,1,0,1,1,-1,0,0,0,0"
TGT[g127]="1,1,1,1,1,1,1,0,0,-1,-1,0,0,0,0"
TGT[g893]="1,0,1,1,1,1,1,0,1,1,-1,0,0,0,-1"
TGT[g1017]="1,0,-2,1,1,1,1,1,1,1,0,0,0,0,0"

for TAG in g885 g127 g893 g1017; do
    SRC=$BASE/results/gr_reduce/$TAG
    DST=$BASE/results/gr_reduce/${TAG}_deltareplay
    rm -rf $DST
    mkdir -p $DST/logs $DST/work/logs $DST/work/results
    cp -l $SRC/work/results/*.pkl $DST/work/results/ 2>/dev/null || \
        cp $SRC/work/results/*.pkl $DST/work/results/
    echo "=== $TAG: replaying $(ls $DST/work/results | wc -l) cached results"
    PYTHONUNBUFFERED=1 $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
        --topology $BASE/topology_input/gravity3L --integral="${TGT[$TAG]}" \
        --output $DST/reduction.pkl --work-dir $DST/work \
        --model-checkpoint $BASE/checkpoints/gravity3L_canon10x_nosubs/best_model.pt \
        --beam_width 40 --max_steps 1000000 --prime 1009 \
        --paper-masters-only --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
        --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
        --max-concurrent 0 --check-interval 1 --use-symmetry --resume \
        > $DST/logs/orch.log 2>&1 || true
    $PYTHON - "$SRC" "$DST" "$TAG" <<'PYEOF'
import pickle, sys
src, dst, tag = sys.argv[1:4]
P = 1009
try:
    a = pickle.load(open(f"{src}/reduction.pkl", "rb"))
    b = pickle.load(open(f"{dst}/reduction.pkl", "rb"))
except FileNotFoundError as e:
    print(f"{tag}: MISSING OUTPUT ({e})"); sys.exit(0)
ea = {tuple(k): v % P for k, v in a["final_expr"].items() if v % P}
eb = {tuple(k): v % P for k, v in b["final_expr"].items() if v % P}
print(f"{tag}: {'IDENTICAL' if ea == eb else 'MISMATCH'} "
      f"({len(ea)} vs {len(eb)} terms)")
PYEOF
done
