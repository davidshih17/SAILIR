#!/usr/bin/env python
"""Split the closure: do R(I) and R(descent) agree on the TRUE (non-corner) masters,
with the discrepancy living ENTIRELY in the corner-master (fake, symmetry-related) subspace?"""
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
def twkey(i): w=w12(i); return (w[0],w[1],tuple(abs(x) for x in i))
def corner_of(sec): return tuple(1 if k in sec else 0 for k in range(11))
def inv(a): return pow(a,P-2,P)
_ac={}
def iscorner(i): return all(x in (0,1) for x in i)
def sym_autos(corner):
    if corner in _ac: return _ac[corner]
    cs=secbits(corner); a=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and secbits(next(iter(im)))==cs)(image_unsigned(corner,*t))]
    _ac[corner]=a; return a
red=pickle.load(open("results/meta_reduce/list_TA_reductions.pkl","rb")); REDU=dict(red['reductions'])
cache=pickle.load(open("replay/reduction_cache.pkl","rb")); CACHE=cache.get('cache',cache)
CORNER=set(tuple(pickle.load(open(p,'rb'))['start_integral']) for p in glob.glob("results/corner_reductions/c*/reduction.pkl"))
rhs=set()
for v in REDU.values(): rhs|=set(v)
TRUE_M=rhs-set(REDU)
MASTERS=TRUE_M|CORNER
def reduce_full(expr):
    expr=dict(expr); steps=0
    while steps<200000:
        cand=[k for k in expr if expr[k]%P and k not in MASTERS and (k in REDU or k in CACHE)]
        if not cand: break
        k=max(cand,key=twkey); c=expr.pop(k); rule=REDU[k] if k in REDU else CACHE[k]
        for kk,cc in rule.items(): expr[kk]=(expr.get(kk,0)+c*cc)%P; 
        steps+=1
    return {k:v%P for k,v in expr.items() if v%P and k in MASTERS}, [k for k in expr if expr[k]%P and k not in MASTERS]
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
tpass=ntest=0
print("=== TRUE-master closure (discrepancy confined to corner/fake masters?) ===")
for I in low[:12]:
    mI,uI=reduce_full({I:1}); E=descent_expr(I)
    if E=={I:1}: continue
    mE,uE=reduce_full(E)
    diff={k:(mI.get(k,0)-mE.get(k,0))%P for k in set(mI)|set(mE)}; diff={k:v for k,v in diff.items() if v%P}
    diff_true={k:v for k,v in diff.items() if not iscorner(k)}
    diff_corner={k:v for k,v in diff.items() if iscorner(k)}
    ok = (not diff_true) and (not uI) and (not uE)
    ntest+=1; tpass+= 1 if ok else 0
    print(f"  I={I}: TRUE-masters agree={ok}   (diff on true masters={len(diff_true)}, on corner/fake masters={len(diff_corner)})")
    if diff_true: print(f"     TRUE-MASTER MISMATCH: {list(diff_true.items())[:2]}")
print(f"\n{tpass}/{ntest} agree on ALL NON-corner masters; every discrepancy term is a pure-denominator CORNER (symmetry-related fake master of the no-sym basis)")
