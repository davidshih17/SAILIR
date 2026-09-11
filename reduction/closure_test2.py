#!/usr/bin/env python
"""CLOSURE: reduce the symmetry-descent expression of I to masters and compare to I's
direct reduction. Fixed: average over APPLICABLE autos (non-None), add identity if absent."""
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
def twkey(i): w=w12(i); return (w[0],w[1],tuple(-abs(x) for x in i))  # LARGER = higher on all 3 components (matches ibp_env.weight) -- consumed as max(cand,key=twkey); with +|abs| the tiebreak picked the LARGEST |abs|, which is backwards
def corner_of(sec): return tuple(1 if k in sec else 0 for k in range(11))
def inv(a): return pow(a,P-2,P)
_ac={}
def sym_autos(corner):
    if corner in _ac: return _ac[corner]
    cs=secbits(corner)
    a=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and secbits(next(iter(im)))==cs)(image_unsigned(corner,*t))]
    _ac[corner]=a; return a
red=pickle.load(open("results/meta_reduce/list_TA_reductions.pkl","rb")); REDU=red['reductions']
cache=pickle.load(open("replay/reduction_cache.pkl","rb")); CACHE=cache.get('cache',cache)
rhs=set(); 
for v in REDU.values(): rhs|=set(v)
MASTERS=rhs-set(REDU)
def reduce_to_masters(expr):
    expr=dict(expr); steps=0
    while True:
        cand=[k for k in expr if expr[k]%P and k not in MASTERS and (k in REDU or k in CACHE)]
        if not cand: break
        k=max(cand,key=twkey); c=expr.pop(k)
        rule=REDU[k] if k in REDU else CACHE[k]
        for kk,cc in rule.items(): expr[kk]=(expr.get(kk,0)+c*cc)%P
        steps+=1
        if steps>500000: break
    unresolved=[k for k in expr if expr[k]%P and k not in MASTERS]
    return {k:v%P for k,v in expr.items() if v%P}, steps, unresolved
def descent_expr(I0):
    E={}; queue=[(I0,1)]; seen=0
    while queue and seen<5000:
        J,cf=queue.pop(0); seen+=1
        autos=sym_autos(corner_of(sector(J)))
        avg={}; napp=0; has_id=False
        for t in autos:
            img=image_unsigned(J,*t)
            if img is None: continue
            napp+=1
            if img=={J:1}: has_id=True
            for k,c in img.items(): avg[k]=avg.get(k,0)+c
        if not has_id: avg[J]=avg.get(J,0)+1; napp+=1
        n=inv(napp); avg={k:(c*n)%P for k,c in avg.items() if c%P}
        wj=w12(J)
        PS={k:(c*cf)%P for k,c in avg.items() if w12(k)==wj}
        R={k:(c*cf)%P for k,c in avg.items() if w12(k)!=wj}
        for k,c in PS.items(): E[k]=(E.get(k,0)+c)%P
        for k,c in R.items(): queue.append((k,c))
    return {k:v for k,v in E.items() if v%P}
ints=[tuple(int(x) for x in re.match(r'TA\[([^\]]+)\]',ln.strip()).group(1).split(','))
      for ln in open("from_federica/list_TA_ispclean_by_weight") if ln.startswith("TA[")]
low=sorted([i for i in ints if i in REDU and len(sym_autos(corner_of(sector(i))))>1 and w12(i)[1]>=1],
           key=lambda i:(w12(i)[0]+w12(i)[1]))
npass=0; ntest=0
for I in low[:8]:
    RI,_,uI=reduce_to_masters({I:1})
    E=descent_expr(I)
    trivial = (E=={I:1})
    RE,steps,uE=reduce_to_masters(E)
    diff={k:(RI.get(k,0)-RE.get(k,0))%P for k in set(RI)|set(RE)}; diff={k:v for k,v in diff.items() if v%P}
    ok = (not diff) and (not uI) and (not uE)
    ntest+=1; npass+= 1 if ok else 0
    print(f"I={I} w={w12(I)}: descent {len(E)} terms{' (TRIVIAL)' if trivial else ''}; "
          f"R(I)={len(RI)}m R(descent)={len(RE)}m unresolved(I/E)={len(uI)}/{len(uE)}  MATCH={ok}")
    if diff: print(f"     DIFF: {list(diff.items())[:3]}")
print(f"\n{npass}/{ntest} closure tests PASS")
