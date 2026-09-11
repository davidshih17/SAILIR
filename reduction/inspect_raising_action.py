#!/usr/bin/env python
"""Which recorded action of the violating g127 worker introduced the
sector-115 term, and is it a derivative IBP or an algebraic den row?
Replays the banked path of async_1446 [2,2,0,-1,1,0,3,...]: for every action
prints target sector, seed sector, op id, and whether the raw equation
contains terms OUTSIDE the start's sector cone (bits beyond sector 83)."""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc

P = 1009
topo = Topology.from_dir(_tc.TOPO_DIR)
ibp_env.init_from_topology(topo)
ibp_env.set_prime(P)
from sailir.ibp_env import get_raw_equation

env = ibp_env.IBPEnvironment()
N = ibp_env.N_INDICES


def sec(t):
    m = 0
    for i in range(_tc.N_DEN):
        if t[i] > 0:
            m |= 1 << i
    return m


pkl = (ROOT + "/results/gr_reduce/g127/work/results/"
       "async_1446_2_2_0_-1_1_0_3_0_0_0_0_0_0_0_0.pkl")
r = pickle.load(open(pkl, "rb"))
start = tuple(r["original_integral"])
S0 = sec(start)
print(f"start {list(start)} sector {S0}; success={r['success']} "
      f"steps={r['steps']}; path length {len(r['path'])}")
print(f"final_expr sectors: "
      f"{sorted({sec(tuple(k)) for k in r['final_expr']})}")

# how many ops are den rows vs derivative IBP vs LI? topology metadata:
print(f"n_actions={topo.n_actions}  (ibp templates: {len(env.ibp_t)}, "
      f"li templates: {len(env.li_t)})")

for i, (target, op, delta) in enumerate(r["path"]):
    target = tuple(target)
    seed = tuple(target[j] + delta[j] for j in range(N))
    raw = get_raw_equation(env.ibp_t, env.li_t, op, seed)
    outside = sorted({sec(tuple(k)) for k in raw
                      if raw[k] % P and (sec(tuple(k)) & ~S0)})
    mark = "  <-- OUTSIDE-CONE terms" if outside else ""
    print(f"  act {i:2d} op={op:2d} target_sec={sec(target):4d} "
          f"seed_sec={sec(seed):4d} raw_terms={len(raw)} "
          f"outside_cone_secs={outside}{mark}")
