#!/usr/bin/env python
"""Trace the symmetry cascade on the target to find the false rule that introduces
the spurious I[0,0,0,0,1,1,0,1,0,0,0] term."""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from symmetry_route import symmetry_rule, tw_key, _build_rules
from canonicalize import _transforms, image_unsigned, P

T=(1,-1,1,0,1,1,0,0,0,0,0)
print(f"target {T}")
# full cascade like the orchestrator
expr={T:1}; cache={}
for step in range(20):
    changed=False
    for I in list(expr):
        if I in cache: continue
        r=symmetry_rule(I)
        if r is not None:
            cache[I]=r; changed=True
    # apply
    new={}
    for I,co in expr.items():
        if I in cache:
            for k,ck in cache[I].items(): new[k]=(new.get(k,0)+co*ck)%P
        else:
            new[I]=(new.get(I,0)+co)%P
    expr={k:v for k,v in new.items() if v%P}
    if not changed: break
print(f"\nfinal cascade terms ({len(expr)}):")
for I,c in sorted(expr.items()): print(f"    {I} = {c}   survivor={symmetry_rule(I) is None}")

# now investigate the target's direct rule + the spurious integral's provenance
print(f"\n=== target's symmetry_rule ===")
r=symmetry_rule(T)
for k,c in sorted(r.items()): print(f"    -> {k} = {c}   (tw lower={tw_key(k)<tw_key(T)})")

SPUR=(0,0,0,0,1,1,0,1,0,0,0)
print(f"\n=== which transform maps some cascade integral onto {SPUR}? ===")
# find every integral I in the closure whose image contains SPUR
for I in [T]+list(r.keys()):
    for (M,c) in _transforms(I):
        img=image_unsigned(I,M,c)
        if img and SPUR in img:
            print(f"    {I}  --sym-->  contains {SPUR} (coeff {img[SPUR]}); full img keys={list(img.keys())}")
