#!/usr/bin/env python
"""Verify the fix: every symmetry rule now sends I to terms strictly LOWER in the
workers' ordering (larger tkey), so it can't cycle with a worker edge. Also confirm
gate_small target still routes and check agreement vs the real m2 worker edges."""
import pickle, sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from symmetry_route import symmetry_rule, tkey

# 1) gate_small target + a few
for I in [(1,-1,1,0,1,1,0,0,0,0,0),(0,-1,2,0,2,-1,0,1,-1,0,0)]:
    r=symmetry_rule(I)
    if r is None: print(f"  {list(I)} -> SURVIVOR"); continue
    ok=all(tkey(k)>tkey(I) for k in r)
    print(f"  {list(I)} -> {len(r)} terms, all strictly-lower-in-ordering={ok}")

# 2) over the REAL m2 baseline integrals: every symmetry rule must be tkey-increasing,
#    AND must NOT reverse any real worker edge (I->J worker) with J->I symmetry.
d=pickle.load(open(BASE+"/results/ab_symmetry/m2_4prop_dots/baseline/reduction.pkl","rb"))
cache=d['cache']
n=len(cache); routed=0; bad_dir=0; reversals=0
# build worker edge set (I -> set of J)
wedges={I:set(cache[I].keys()) for I in cache}
sym={}
for I in cache:
    r=symmetry_rule(I)
    if r is None: continue
    routed+=1; sym[I]=set(r.keys())
    if not all(tkey(k)>tkey(I) for k in r): bad_dir+=1
# reversal check: is there I with worker J in cache[I], and symmetry rule J->...I...?
for I in wedges:
    for J in wedges[I]:
        if J in sym and I in sym[J]:
            reversals+=1
print(f"\nover {n} real m2 integrals: {routed} get a symmetry rule")
print(f"  symmetry rules NOT strictly-lower-in-ordering : {bad_dir}   (must be 0)")
print(f"  worker I->J with symmetry J->...I... (cycles)  : {reversals}   (must be 0)")
