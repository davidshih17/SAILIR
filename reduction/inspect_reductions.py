#!/usr/bin/env python
import sys, os, pickle
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
for p in ["results/meta_reduce/list_TA_reductions.pkl","replay/reduction_cache.pkl"]:
    print(f"########## {p} ##########")
    try:
        with open(p,'rb') as f: d=pickle.load(f)
    except Exception as e:
        print("  load err:", e); continue
    print("  type:", type(d).__name__, " len:", len(d) if hasattr(d,'__len__') else '')
    if isinstance(d,dict):
        # maybe wrapped
        if 'cache' in d: d=d['cache']; print("  (unwrapped 'cache') len:", len(d))
        ks=list(d.keys())[:2]
        print("  key type:", type(ks[0]).__name__ if ks else None, " sample key:", ks[0] if ks else None)
        for k in ks:
            v=d[k]
            print(f"    {k} -> {type(v).__name__} len={len(v) if hasattr(v,'__len__') else ''}")
            if isinstance(v,dict):
                for kk,vv in list(v.items())[:3]: print(f"        {kk}: {str(vv)[:60]}")
            else:
                print(f"        {str(v)[:120]}")
    print()
