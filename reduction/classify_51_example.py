#!/usr/bin/env python
"""For the (5,1) example I=(0,1,1,0,1,-1,0,2,0,0,0) [sector {1,2,4,7}], take every
symmetry image and classify EVERY term by sector (prop-SET) and weight, relative to I.
Question: how many image terms are in a LOWER sector (dropped in reduction) vs same
sector same weight (the genuine mixing that survives)?"""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
ND=8
def props(i):  return frozenset(k for k in range(ND) if i[k]>0)
def pcount(i): return sum(1 for k in range(ND) if i[k]>0)
def wt(i):     return (sum(x for x in i if x>0), sum(-x for x in i if x<0))

I=(0,1,1,0,1,-1,0,2,0,0,0)
Sset=props(I); wI=wt(I); pcI=pcount(I)
print(f"I = {I}")
print(f"  sector(prop-set) = {sorted(Sset)}   prop-count = {pcI}   weight = {wI}\n")

# gather all symmetry images of I
imgs=[]
for t in _transforms(I):
    im=image_unsigned(I,*t)
    if im and len(im)>1:            # nontrivial (multi-term) images
        imgs.append(im)

def classify(k):
    pk, wk = props(k), wt(k)
    if pk==Sset:
        return "SAME-sector same-w" if wk==wI else f"SAME-sector lower-w {wk}"
    if pk < Sset:  return f"SUB-sector({len(pk)}p)  <-- DROPPED"
    if pk > Sset:  return f"SUPER-sector({len(pk)}p)"
    return f"OTHER-sector({sorted(pk)})"

# find the image with the most SAME-sector-same-weight terms (the mixing case)
best=None; bestn=-1
for im in imgs:
    n=sum(1 for k in im if props(k)==Sset and wt(k)==wI)
    if n>bestn: bestn=n; best=im
print(f"most-mixing image has {bestn} same-sector same-weight terms; full breakdown:\n")
from collections import Counter
cat=Counter()
for k,c in sorted(best.items()):
    cl=classify(k)
    cat[cl.split(' <--')[0].split(' {')[0]]+=1
    print(f"  {k}  coeff={c%1009:>4}  sector={sorted(props(k))} w={wt(k)}   {cl}")
print("\nsummary of this image:")
for cl,n in cat.most_common(): print(f"  {n:2d}  {cl}")

# aggregate across ALL images: term-sector histogram
print("\n=== aggregate over ALL", len(imgs), "nontrivial images of I ===")
agg=Counter()
for im in imgs:
    for k in im:
        pk=props(k)
        if pk==Sset:  agg["same-sector same-w" if wt(k)==wI else "same-sector lower-w"]+=1
        elif pk<Sset: agg["SUB-sector (dropped)"]+=1
        elif pk>Sset: agg["super-sector"]+=1
        else:         agg["other-sector"]+=1
tot=sum(agg.values())
for cl,n in agg.most_common(): print(f"  {n:4d} ({100*n/tot:4.1f}%)  {cl}")
