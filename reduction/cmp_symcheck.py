#!/usr/bin/env python
"""Post-strip regression compare: for each *_symcheck design1 run, check masters +
worker count against the STORED pre-strip runs. Nothing broke iff the new design1
masters match the old (baseline and/or pre-strip design1) and the worker count
reproduces the old savings."""
import sys, os, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE)
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE, "topology_input/pentagonbox")))
ibp_env.set_prime(1009)
ROOT = os.path.join(BASE, "results/ab_symmetry")

# new _symcheck tag -> old stored tag
PAIR = {
    "gate_small_symcheck": "gate_small",
    "m1_symcheck":         "m1_6prop",
    "m2_symcheck":         "m2_4prop_dots",
    "m3_symcheck":         "m3_5prop_deg3",
}

def load(tag, arm):
    p = os.path.join(ROOT, tag, arm, "reduction.pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None

def masters(r):
    return {i: c for i, c in r['final_expr'].items() if c != 0} if r else None

print(f"{'target':<20} {'new_wk':>7} {'old_d1_wk':>9} {'old_base_wk':>11}  {'masters vs old-design1':<24} {'masters vs old-baseline'}")
print("-" * 110)
for new, old in PAIR.items():
    nd = load(new, "design1")
    if nd is None:
        print(f"{new:<20} {'RUNNING (no reduction.pkl yet)':>7}")
        continue
    mn = masters(nd)
    od = load(old, "design1"); ob = load(old, "baseline")
    md = masters(od); mb = masters(ob)
    vs_d1 = "n/a (no old design1)" if md is None else ("MATCH ✓" if mn == md else "*** DIFFER ***")
    vs_b  = "n/a (no old baseline)" if mb is None else ("MATCH ✓" if mn == mb else "*** DIFFER ***")
    print(f"{new:<20} {nd['total_jobs']:>7} {(od['total_jobs'] if od else '-'):>9} "
          f"{(ob['total_jobs'] if ob else '-'):>11}  {vs_d1:<24} {vs_b}   "
          f"[{len(mn)} masters, {nd.get('n_symmetry_routed',0)} routed free]")
