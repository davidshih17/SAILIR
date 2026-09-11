#!/usr/bin/env python
"""Pin the cycle: are cache edges strictly-decreasing in the WORKERS' weight order?
Compare baseline (all-worker) vs design1 (worker+symmetry). Identify the bad edge."""
import pickle, sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from sailir.ibp_env import weight            # the worker/orchestrator weight order
from symmetry_route import symmetry_rule, tw_key

def check(tag):
    d=pickle.load(open(f"{BASE}/results/ab_symmetry/m2_4prop_dots/{tag}/reduction.pkl","rb"))
    cache=d['cache']; edges=0; bad=0; badex=[]
    for I,red in cache.items():
        wI=weight(I)
        for J in red:
            edges+=1
            if not (weight(J) < wI):        # not strictly lower in worker order
                bad+=1
                if len(badex)<4: badex.append((I,J))
    print(f"[{tag}] {len(cache)} cache entries, {edges} edges; NOT-strictly-lower(worker weight): {bad}")
    return cache, badex

cb,_=check("baseline")
cd,badex=check("design1")
print("\n=== design1 weight-increasing / lateral edges (the cycle culprits) ===")
for I,J in badex:
    wI,wJ=weight(I),weight(J)
    # provenance: does symmetry_rule reproduce cache[I]?
    sr=symmetry_rule(I)
    prov = "SYMMETRY" if (sr is not None and set(sr)==set(cd[I])) else ("SURVIVOR->worker" if sr is None else "WORKER(or mixed)")
    print(f"  edge I{list(I)} -> J{list(J)}")
    print(f"     weight(I)={wI[:2]},{wI[2]}   weight(J)={wJ[:2]},{wJ[2]}   J<I:{wJ<wI}")
    print(f"     cache[I] provenance: {prov}  (symmetry_rule(I) is {'None' if sr is None else str(len(sr))+' terms'})")
