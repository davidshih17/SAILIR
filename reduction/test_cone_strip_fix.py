#!/usr/bin/env python
"""Unit test of the out-of-cone strip exemption + filter rejection, on the
exact action that poisoned g127 (async_1446, act 1: op 6, target sector 83,
seed sector 115)."""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc

P = 1009
ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
ibp_env.set_prime(P)
from sailir.ibp_env import (get_raw_equation, action_introduces_outside_sector,
                            weight)

env = ibp_env.IBPEnvironment()
N = ibp_env.N_INDICES

r = pickle.load(open(ROOT + "/results/gr_reduce/g127/work/results/"
                     "async_1446_2_2_0_-1_1_0_3_0_0_0_0_0_0_0_0.pkl", "rb"))
start = tuple(r["original_integral"])
w = weight(start)
start_w12 = (w[0], w[1])
target, op, delta = r["path"][1]
target = tuple(target)
seed = tuple(target[i] + delta[i] for i in range(N))
print(f"start {list(start)} w12={start_w12}; offending action op={op} "
      f"target={list(target)} seed={list(seed)}")


def sec(t):
    return sum(1 << i for i in range(_tc.N_DEN) if t[i] > 0)


cone = sec(start)

# OLD behavior: strip without cone
ibp_env.set_raw_strip_cone(None)
raw_old = get_raw_equation(env.ibp_t, env.li_t, op, seed, min_w12=start_w12)
out_old = [k for k in raw_old if sec(k) & ~cone]
# NEW behavior: cone exemption
ibp_env.set_raw_strip_cone(cone)
raw_new = get_raw_equation(env.ibp_t, env.li_t, op, seed, min_w12=start_w12)
out_new = [k for k in raw_new if sec(k) & ~cone]
rejected = action_introduces_outside_sector(raw_new, target)
print(f"OLD stripped eq: {len(raw_old)} terms, out-of-cone visible: {len(out_old)}")
print(f"NEW stripped eq: {len(raw_new)} terms, out-of-cone visible: {len(out_new)}")
for k in out_new[:3]:
    print(f"   kept out-of-cone term: {list(k)} (sector {sec(k)})")
print(f"subsector filter now rejects the action: {rejected}")
# unstripped reference
raw_full = get_raw_equation(env.ibp_t, env.li_t, op, seed)
print(f"unstripped eq: {len(raw_full)} terms "
      f"(NEW == restriction consistent: "
      f"{all(k in raw_full and raw_full[k] % P == raw_new[k] % P for k in raw_new)})")
ok = (len(out_old) == 0 and len(out_new) > 0 and rejected)
print("PASS" if ok else "FAIL")
