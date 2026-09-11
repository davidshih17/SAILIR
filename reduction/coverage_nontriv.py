#!/usr/bin/env python
"""Integral-level clean-orbit coverage among NON-TRIVIAL sectors only (the data-gen
relevant ones). If low, training-side canonicalization buys little for pentagon-box."""
import sys, os, itertools
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from canonical_rep import clean_orbit
TRIV=set(int(x) for x in open(BASE+"/results/kira_reduce_161/sectormappings/TA/trivialsector").read().split(",") if x.strip())
ISP=[8,9,10]
nontriv=[m for m in range(1,256) if m not in TRIV]
def corner(mask): return tuple(1 if k in [j for j in range(8) if mask>>j&1] else 0 for k in range(11))
for deg in [0,1,2]:
    n=nn=0
    for m in nontriv:
        base=corner(m)
        combos=[()] if deg==0 else list(itertools.combinations_with_replacement(ISP,deg))
        for cp in combos:
            I=list(base)
            for s in cp: I[s]-=1
            orb=clean_orbit(tuple(I))
            if orb is None: continue
            n+=1
            if len(orb)>1: nn+=1
    lbl={0:"corners",1:"deg-1 numerators",2:"deg-2 numerators"}[deg]
    print(f"  {lbl:20s} in non-trivial sectors: {nn}/{n} have orbit>1 ({100*nn/max(1,n):.0f}%)")
