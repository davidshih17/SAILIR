#!/usr/bin/env python
"""FINAL closure: merge the freshly-reduced corners into the reduction table, then verify
symmetry descent of I reduces to the SAME masters as I's direct reduction."""
import sys, os, pickle, re, glob
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def secbits(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
def sector(i): return tuple(k for k in range(ND) if i[k]>0)
def w12(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0))
def twkey(i): w=w12(i); return (w[0],w[1],tuple(-abs(x) for x in i))  # LARGER = higher on all 3 components (matches ibp_env.weight) -- consumed as max(cand,key=twkey); with +|abs| the tiebreak picked the LARGEST |abs|, which is backwards
def corner_of(sec): return tuple(1 if k in sec else 0 for k in range(11))
def inv(a): return pow(a,P-2,P)
_ac={}
def sym_autos(corner):
    if corner in _ac: return _ac[corner]
    cs=secbits(corner); a=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and secbits(next(iter(im)))==cs)(image_unsigned(corner,*t))]
    _ac[corner]=a; return a

red=pickle.load(open("results/meta_reduce/list_TA_reductions.pkl","rb")); REDU=dict(red['reductions'])
cache=pickle.load(open("replay/reduction_cache.pkl","rb")); CACHE=cache.get('cache',cache)

# merge corner reductions
def load_corner_red(pth):
    d=pickle.load(open(pth,'rb'))
    for key in ('reduction','final_expr','final_masters','masters','result'):
        if key in d and isinstance(d[key],dict): return d[key]
    # else find the tuple->dict mapping
    for k,v in d.items():
        if isinstance(k,tuple) and isinstance(v,dict): return {k:v}
    return None
CORNER_RED={}
for pth in sorted(glob.glob("results/corner_reductions/c*/reduction.pkl")):
    d=pickle.load(open(pth,'rb'))
    # the target is the reduced integral; result maps it to masters
    tgt=d.get('start_integral') or d.get('integral') or d.get('target')
    expr=d.get('final_expr') or d.get('reduction') or d.get('masters') or d.get('final_masters')
    if tgt and isinstance(expr,dict):
        CORNER_RED[tuple(tgt)]={tuple(k):v for k,v in expr.items()}
    else:
        # fallback: probe keys once
        if not CORNER_RED: print("  corner pkl keys:", list(d.keys()))
print(f"loaded {len(CORNER_RED)} corner reductions")
for c,v in list(CORNER_RED.items())[:3]: print(f"   {c} -> {len(v)} masters: {list(v.items())[:2]}")

EXT={**REDU, **CORNER_RED}
rhs=set()
for v in EXT.values(): rhs|=set(v)
MASTERS=rhs-set(EXT)
print(f"extended reduction table: {len(EXT)} entries, {len(MASTERS)} masters")

def reduce_full(expr):
    expr=dict(expr)
    while True:
        cand=[k for k in expr if expr[k]%P and k not in MASTERS and (k in EXT or k in CACHE)]
        if not cand: break
        k=max(cand,key=twkey); c=expr.pop(k); rule=EXT[k] if k in EXT else CACHE[k]
        for kk,cc in rule.items(): expr[kk]=(expr.get(kk,0)+c*cc)%P
    expr={k:v%P for k,v in expr.items() if v%P}
    return {k:v for k,v in expr.items() if k in MASTERS}, {k:v for k,v in expr.items() if k not in MASTERS}
def descent_expr(I0):
    E={}; queue=[(I0,1)]; seen=0
    while queue and seen<5000:
        J,cf=queue.pop(0); seen+=1
        autos=sym_autos(corner_of(sector(J))); avg={}; napp=0; has_id=False
        for t in autos:
            img=image_unsigned(J,*t)
            if img is None: continue
            napp+=1
            if img=={J:1}: has_id=True
            for k,c in img.items(): avg[k]=avg.get(k,0)+c
        if not has_id: avg[J]=avg.get(J,0)+1; napp+=1
        n=inv(napp); avg={k:(c*n)%P for k,c in avg.items() if c%P}; wj=w12(J)
        for k,c in avg.items():
            if w12(k)==wj: E[k]=(E.get(k,0)+(c*cf))%P
            else: queue.append((k,(c*cf)%P))
    return {k:v for k,v in E.items() if v%P}

ints=[tuple(int(x) for x in re.match(r'TA\[([^\]]+)\]',ln.strip()).group(1).split(','))
      for ln in open("from_federica/list_TA_ispclean_by_weight") if ln.startswith("TA[")]
low=sorted([i for i in ints if i in REDU and len(sym_autos(corner_of(sector(i))))>1 and w12(i)[1]>=1],
           key=lambda i:(w12(i)[0]+w12(i)[1]))
npass=ntest=0
print("\n=== CLOSURE (descent-to-masters == direct reduction) ===")
for I in low[:12]:
    mI,uI=reduce_full({I:1})
    E=descent_expr(I)
    if E=={I:1}: continue
    mE,uE=reduce_full(E)
    diff={k:(mI.get(k,0)-mE.get(k,0))%P for k in set(mI)|set(mE)}; diff={k:v for k,v in diff.items() if v%P}
    ok=(not diff) and (not uI) and (not uE)
    ntest+=1; npass+= 1 if ok else 0
    print(f"  I={I}: MATCH={ok}  R(I)={len(mI)}m R(desc)={len(mE)}m unresolved(I/E)={len(uI)}/{len(uE)}"
          + ("" if ok else f"  DIFF={list(diff.items())[:2]}"))
print(f"\n{npass}/{ntest} closure tests PASS (symmetry descent agrees with orchestrator reduction)")
