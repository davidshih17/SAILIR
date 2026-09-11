#!/usr/bin/env python
"""Is a placeholder-UNIQUE reduction correct? Take an integral the placeholder makes 'free',
form the symmetry relation R = I - sigma(I), reduce R to masters. If the placeholder is a
real relation, R -> 0. If it's the fake permutation, R -> nonzero."""
import sys, os, pickle
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
import glob
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

I=(1,1,1,1,1,2,0,0,-3,0,0)
cs=secbits(corner_of(sector(I)))
# find the placeholder auto (partial-M, non-None on I) that full-M autos can't match
for (M,c) in _transforms(corner_of(sector(I))):
    im=image_unsigned(corner_of(sector(I)),M,c)
    if not (im and len(im)==1 and secbits(next(iter(im)))==cs): continue
    if all(i in M for i in range(N)): continue     # skip full-M
    sig=image_unsigned(I,M,c)
    if sig is None or sig=={I:1}: continue
    print(f"placeholder auto applies to I={I}: sigma(I) has {len(sig)} terms")
    R={I:1}
    for k,co in sig.items(): R[k]=(R.get(k,0)-co)%P
    R={k:co for k,co in R.items() if co%P}
    mR,unres=reduce_full(R)
    print(f"  R = I - sigma(I) reduced to masters: {mR}")
    print(f"  unresolved (not in cache): {len(unres)}")
    if not mR and not unres: print("  => R = 0  ==> placeholder relation is CORRECT")
    elif not unres: print(f"  => R = {len(mR)} nonzero master terms  ==> placeholder relation is WRONG (fake permutation)")
    else: print(f"  => inconclusive: {len(unres)} terms not in cache (would need orchestrator)")
    break
