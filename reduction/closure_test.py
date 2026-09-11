#!/usr/bin/env python
"""CLOSURE TEST: symmetry descent gives I = P_S.I + Σ c_j C_j (exact). Reduce that whole
RHS to masters (via list_TA_reductions + one-step cache) and compare to I's direct
reduction R(I). They must match if the symmetry descent is consistent with the IBP reduction."""
import sys, os, pickle, re
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
def twkey(i): w=w12(i); return (w[0],w[1],tuple(-abs(x) for x in i))  # LARGER = higher on all 3 components (matches ibp_env.weight) -- consumed as max(cand,key=twkey); with +|abs| the tiebreak picked the LARGEST |abs|, which is backwards  # total weight, higher=bigger
def corner_of(sec): return tuple(1 if k in sec else 0 for k in range(11))
def inv(a): return pow(a,P-2,P)
_ac={}
def sym_autos(corner):
    if corner in _ac: return _ac[corner]
    cs=secbits(corner)
    a=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and secbits(next(iter(im)))==cs)(image_unsigned(corner,*t))]
    _ac[corner]=a; return a

# load reductions
red=pickle.load(open("results/meta_reduce/list_TA_reductions.pkl","rb"))
print("list_TA_reductions keys:", list(red.keys()))
REDU = red.get('reductions') or red.get('reduction') or {k:v for k,v in red.items() if isinstance(k,tuple)}
print("  #reductions:", len(REDU), " sample:", next(iter(REDU)) if REDU else None)
cache=pickle.load(open("replay/reduction_cache.pkl","rb"))
CACHE=cache.get('cache',cache)
print("  one-step cache size:", len(CACHE))
# master set = all integrals appearing on RHS of REDU that are NOT themselves keys
rhs=set(); lhs=set(REDU.keys())
for v in REDU.values():
    for k in v: rhs.add(k)
MASTERS = rhs - lhs
print("  distinct masters (rhs-lhs):", len(MASTERS))

def reduce_to_masters(expr, budget=200000):
    """iteratively expand any non-master term using REDU (to-masters) or CACHE (one-step)."""
    expr=dict(expr); steps=0
    while True:
        # find highest-total-weight non-master term with a rule
        cand=[k for k in expr if expr[k]%P and k not in MASTERS]
        cand=[k for k in cand if k in REDU or k in CACHE]
        if not cand: break
        k=max(cand,key=twkey); c=expr.pop(k)
        rule = REDU[k] if k in REDU else CACHE[k]
        for kk,cc in rule.items(): expr[kk]=(expr.get(kk,0)+c*cc)%P
        steps+=1
        if steps>budget: print("  BUDGET HIT"); break
    return {k:v%P for k,v in expr.items() if v%P}, steps

# symmetry descent of I -> expression E (standard integrals)
def descent_expr(I0):
    E={}; queue=[(I0,1)]; seen=0
    while queue and seen<2000:
        J,cf=queue.pop(0); seen+=1
        autos=sym_autos(corner_of(sector(J)))
        if len(autos)<=1: E[J]=(E.get(J,0)+cf)%P; continue
        avg={}
        for t in autos:
            img=image_unsigned(J,*t)
            if img is None: continue
            for k,c in img.items(): avg[k]=avg.get(k,0)+c
        n=inv(len(autos)); avg={k:(c*n)%P for k,c in avg.items() if c%P}
        wj=w12(J)
        R={k:(c*cf)%P for k,c in avg.items() if w12(k)!=wj}
        PS={k:(c*cf)%P for k,c in avg.items() if w12(k)==wj}
        if not R:  # terminal
            for k,c in PS.items(): E[k]=(E.get(k,0)+c)%P
            continue
        for k,c in PS.items(): E[k]=(E.get(k,0)+c)%P
        for k,c in R.items(): queue.append((k,c))
    return {k:v for k,v in E.items() if v%P}

# pick a list_TA integral in a symmetric sector that IS in REDU
ints=[tuple(int(x) for x in re.match(r'TA\[([^\]]+)\]',ln.strip()).group(1).split(','))
      for ln in open("from_federica/list_TA_ispclean_by_weight") if ln.startswith("TA[")]
cands=[i for i in ints if i in REDU and len(sym_autos(corner_of(sector(i))))>1 and w12(i)[1]>=1]
for I in cands[:3]:
    print(f"\n===== CLOSURE for I={I} sector {list(sector(I))} w={w12(I)} =====")
    RI,_=reduce_to_masters({I:1})
    E=descent_expr(I)
    print(f"  descent expression E has {len(E)} standard-integral terms")
    RE,steps=reduce_to_masters(E)
    diff={k:(RI.get(k,0)-RE.get(k,0))%P for k in set(RI)|set(RE)}
    diff={k:v for k,v in diff.items() if v%P}
    print(f"  R(I) has {len(RI)} masters; R(descent) has {len(RE)} masters; reduce steps={steps}")
    print(f"  MATCH: {not diff}" + ("" if not diff else f"   DIFF terms: {len(diff)} e.g. {list(diff.items())[:3]}"))
