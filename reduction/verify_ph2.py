#!/usr/bin/env python
"""Find a cache integral that a PLACEHOLDER (partial-M) auto directly applies to and reduces,
then test whether that relation R = I - sigma(I) actually holds (reduces to 0)."""
import sys, os, pickle, glob
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8; N=11
def secbits(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
def sector(i): return tuple(k for k in range(ND) if i[k]>0)
def corner_of(sec): return tuple(1 if k in sec else 0 for k in range(11))
def w12(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0))
def twkey(i): w=w12(i); return (w[0],w[1],tuple(abs(x) for x in i))
def inv(a): return pow(a,P-2,P)
red=pickle.load(open("results/meta_reduce/list_TA_reductions.pkl","rb")); REDU=red['reductions']
cache=pickle.load(open("replay/reduction_cache.pkl","rb")); CACHE=cache.get('cache',cache)
CORNER=set(tuple(pickle.load(open(p,'rb'))['start_integral']) for p in glob.glob("results/corner_reductions/c*/reduction.pkl"))
rhs=set()
for v in REDU.values(): rhs|=set(v)
MASTERS=(rhs-set(REDU))|CORNER
def reduce_full(expr):
    expr=dict(expr); steps=0
    while steps<300000:
        cand=[k for k in expr if expr[k]%P and k not in MASTERS and (k in REDU or k in CACHE)]
        if not cand: break
        k=max(cand,key=twkey); c=expr.pop(k); rule=REDU[k] if k in REDU else CACHE[k]
        for kk,cc in rule.items(): expr[kk]=(expr.get(kk,0)+c*cc)%P
        steps+=1
    return {k:v%P for k,v in expr.items() if v%P and k in MASTERS}, [k for k in expr if expr[k]%P and k not in MASTERS]
_ac={}
def autos(corner):
    if corner in _ac: return _ac[corner]
    cs=secbits(corner); a=[(M,c) for (M,c) in _transforms(corner) if (lambda im: im and len(im)==1 and secbits(next(iter(im)))==cs)(image_unsigned(corner,M,c))]
    _ac[corner]=a; return a
# find integrals where a partial-M auto applies (non-None, non-identity) and gives a real relation to test
tested=0
for I in CACHE:
    if tested>=3: break
    for (M,c) in autos(corner_of(sector(I))):
        if all(i in M for i in range(N)): continue    # skip full-M
        sig=image_unsigned(I,M,c)
        if sig is None or sig=={I:1}: continue
        R={I:1}
        for k,co in sig.items(): R[k]=(R.get(k,0)-co)%P
        R={k:co for k,co in R.items() if co%P}
        mR,unres=reduce_full(R)
        if unres: continue   # need all covered to judge
        verdict = "CORRECT (R->0)" if not mR else f"WRONG (R->{len(mR)} nonzero masters)"
        print(f"I={I}: placeholder relation sigma(I) has {len(sig)} terms -> {verdict}")
        if mR: print(f"     R residual: {list(mR.items())[:3]}")
        tested+=1
        break
if tested==0: print("no directly-testable placeholder relation found in cache sample")
