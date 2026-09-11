#!/usr/bin/env python
import sys, os, pickle
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
p=sys.argv[1]
with open(p,'rb') as f: d=pickle.load(f)
print("type:", type(d))
if isinstance(d, dict):
    for k,v in d.items():
        t=type(v).__name__; n = len(v) if hasattr(v,'__len__') else ''
        print(f"  {k!r}: {t}  len={n}")
    for k in ('full_subs_replay','full_expr_replay','resolved_subs'):
        if k in d and isinstance(d[k],dict):
            ks=list(d[k].keys())[:3]
            print(f"\n  sample keys of {k!r} ({type(ks[0]).__name__}):")
            for kk in ks: print(f"     {kk!r}")
