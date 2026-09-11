#!/usr/bin/env python
"""Show concrete cross-sector (lateral) symmetry relations: integral before, terms
after, with sectors labeled. A 'lateral' term has the SAME prop-count but a DIFFERENT
prop-set than the source integral."""
import sys, os, itertools
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
topo = Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox_nosym"))
ibp_env.init_from_topology(topo); ibp_env.set_prime(1009)
from symmetry_engine import N
from canonicalize import _transforms, image_unsigned
ND=8
def props(i): return tuple(k for k in range(ND) if i[k]>0)
def npr(i): return len(props(i))
def w(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0))
def classify(src, k):
    ps, pk = set(props(src)), set(props(k))
    if pk==ps: return "same-sector"
    if pk < ps: return "SUBSECTOR (downward)"
    if len(pk)==len(ps): return "*** LATERAL (different same-count sector) ***"
    if len(pk)>len(ps): return "HIGHER"
    return f"subset-diff({len(pk)}p)"

def show(I, want_lateral=True, want_multi=False):
    for (M,c) in _transforms(I):
        img = image_unsigned(I, M, c)
        if img is None: continue
        if img == {I:1}: continue           # trivial identity
        has_lat = any(classify(I,k).startswith('***') for k in img)
        is_multi = len(img) > 1
        if (want_lateral and has_lat) or (want_multi and is_multi):
            print(f"\nBEFORE: I = {I}")
            print(f"        props={props(I)} ({npr(I)} props)  weight={w(I)}")
            print(f"AFTER (image = {len(img)} term(s)):")
            for k in sorted(img):
                print(f"   {k}  coeff={img[k]}")
            for k in sorted(img):
                print(f"     props({k}) = {props(k)}  ->  {classify(I,k)}")
            return True
    return False

print("="*70); print("EXAMPLE 1: a pure DENOMINATOR, cross-sector (lateral) relabeling")
print("="*70)
for np_ in (3,4):
    done=False
    for pr in itertools.combinations(range(ND),np_):
        b=[0]*N
        for p in pr: b[p]=1
        if show(tuple(b), want_lateral=True):
            done=True; break
    if done: break

print("\n"+"="*70); print("EXAMPLE 2: a NUMERATOR integral, multi-term image (ISP<->prop mixing)")
print("="*70)
for I in [(0,1,1,0,1,-1,0,1,0,0,0),(0,1,1,0,1,0,0,1,-1,0,0),(1,1,1,0,0,-1,0,0,0,0,0)]:
    if show(I, want_lateral=False, want_multi=True): break

print("\n"+"="*70)
print("SEARCH: is there a relation that is BOTH lateral AND multi-term?")
print("(i.e. a nontrivial cross-sector move — the case that would break IBP flow)")
print("="*70)
found=0
import itertools as it
cands=[]
for np_ in (3,4):
    for pr in it.combinations(range(ND),np_):
        b=[0]*N
        for p in pr: b[p]=1
        cands.append(tuple(b))
        for s in (8,9,10):
            t=list(b); t[s]=-1; cands.append(tuple(t))
        for j in range(ND):
            if j not in pr:
                t=list(b); t[j]=-1; cands.append(tuple(t)); break
tot_lateral=0; tot_multi=0; both=0
for I in cands:
    for (M,c) in _transforms(I):
        img=image_unsigned(I,M,c)
        if img is None or img=={I:1}: continue
        lat=[k for k in img if classify(I,k).startswith('***')]
        multi=len(img)>1
        if lat: tot_lateral+=1
        if multi: tot_multi+=1
        if lat and multi:
            both+=1
            if found<3:
                found+=1
                print(f"\n  FOUND lateral+multi: I={I} props={props(I)}")
                for k in sorted(img):
                    print(f"     {k} props={props(k)} coeff={img[k]}  {classify(I,k)}")
print(f"\ntotals over {len(cands)} integrals' transforms:")
print(f"  relations with a LATERAL term : {tot_lateral}")
print(f"  relations that are MULTI-term : {tot_multi}")
print(f"  relations that are BOTH lateral AND multi-term : {both}")
