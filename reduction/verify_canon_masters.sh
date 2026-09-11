#!/bin/bash
# Correctness gate for canon+collect: run the SAME single-step reduction of the
# corner Z=(0,1,1,0,1,0,0,1,0,0,0) WITHOUT canon (reference, drains in ~3 steps) and
# WITH canon (large budget). Then compare the final master combinations. If canon
# converges AND its masters match the no-canon masters, the canon+collect fold-back
# handling is value-correct and the extra steps are purely model/nav overhead.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=$BASE/../RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=$BASE/topology_input/pentagonbox_nosym
INT='0,1,1,0,1,0,0,1,0,0,0'

common=(--topology "$TOPO" --model "$MODEL" --integral="$INT"
        --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40
        --max-actions 900 --beam-sort weight --paper-masters-only --prime 1009
        --n-threads 1 --n-workers 1 --device cpu --model-batch-chunk 8)

export PYTHONUNBUFFERED=1 MALLOC_MMAP_THRESHOLD_=67108864 SAILIR_END_OF_STEP_TRIM=1 \
       SAILIR_TABU_CAP=0 SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 SAILIR_SUCCESS_TOTAL=1 \
       SAILIR_ACTION_SELECT=maxweight

# --- reference: no canon ---
DR=$BASE/results/verify_nocanon; rm -rf "$DR"; mkdir -p "$DR"
SAILIR_CANON=0 $PY -u $BASE/reduction/beam_search_v7.py "${common[@]}" \
    --output "$DR/result.pkl" --ckpt "$DR/ckpt.pkl" --ckpt-every 100000 \
    --max-steps 40 > "$DR/out.log" 2>&1
echo "NOCANON exit=$? -> $DR/out.log"

# --- canon+collect, large budget ---
DC=$BASE/results/verify_canon; rm -rf "$DC"; mkdir -p "$DC"
SAILIR_CANON=1 $PY -u $BASE/reduction/beam_search_v7.py "${common[@]}" \
    --output "$DC/result.pkl" --ckpt "$DC/ckpt.pkl" --ckpt-every 100000 \
    --max-steps 400 > "$DC/out.log" 2>&1
echo "CANON exit=$? -> $DC/out.log"

# --- compare masters ---
$PY - "$DR/result.pkl" "$DC/result.pkl" <<'PYEOF'
import pickle, sys
def load(p):
    try:
        with open(p,'rb') as f: return pickle.load(f)
    except Exception as e:
        return {'_err': repr(e)}
r = load(sys.argv[1]); c = load(sys.argv[2])
def summ(tag, d):
    print(f"=== {tag} ===")
    if not isinstance(d, dict): print("  (not a dict):", type(d)); return None
    if '_err' in d: print("  load err:", d['_err']); return None
    print("  keys:", list(d.keys())[:20])
    for k in ('success','n_steps','masters','final_expr','master_coeffs'):
        if k in d:
            v = d[k]
            print(f"  {k}: {v if not isinstance(v,(dict,list)) or len(str(v))<200 else str(v)[:200]+'...'}")
    return d
rr = summ("NOCANON", r); cc = summ("CANON", c)
def masters(d):
    if not isinstance(d, dict): return None
    for k in ('masters','master_coeffs','final_expr'):
        if k in d and isinstance(d[k], dict): return d[k]
    return None
mr, mc = masters(rr), masters(cc)
if mr is not None and mc is not None:
    same = (mr == mc)
    print("\n=== MASTER COMPARISON ===")
    print("  identical:", same)
    if not same:
        kr, kc = set(mr), set(mc)
        print("  only in NOCANON:", list(kr-kc)[:10])
        print("  only in CANON  :", list(kc-kr)[:10])
        for k in (kr & kc):
            if mr[k]!=mc[k]: print("  coeff diff", k, mr[k], mc[k])
else:
    print("\n(could not extract comparable master dicts from one/both results)")
PYEOF
