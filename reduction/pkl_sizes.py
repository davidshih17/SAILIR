#!/usr/bin/env python
import sys, os, pickle
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
for p in sys.argv[1:]:
    try:
        with open(p,'rb') as f: d=pickle.load(f)
        subs=len(d.get('full_subs_replay',{})); expr=len(d.get('full_expr_replay',{}))
        start=d.get('start_integral'); nsteps=d.get('best_state',{}).get('n_steps','?') if isinstance(d.get('best_state'),dict) else '?'
        print(f"  subs={subs:5d}  expr={expr:5d}  start={start}   {p}")
    except Exception as e:
        print(f"  ERR {p}: {e}")
