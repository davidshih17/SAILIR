#!/usr/bin/env python
"""Altitude profile of the truth one-step paths for the two stuck g885
integrals: for each certified action (in dependency order) the solved pivot's
(r,s), plus a simulated worker replay tracking the ACTIVE expression's max
weight and size step by step. This is the search-corridor spec the beam
hyperparameters must accommodate."""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
import topo_config as _tc
from symmetry_route import tkey

P = 1009


def rs(t):
    return (sum(x for x in t if x > 0), sum(-x for x in t if x < 0))


with open(os.path.join(ROOT, "results/truth/g885_stuck2_onestep.pkl"), "rb") as f:
    data = pickle.load(f)

for T, r in data.items():
    if r is None:
        continue
    acts = r["actions"]
    sol = r["sol_store"]
    print(f"\n===== {list(T)}  start rs={rs(T)}  {len(acts)} actions, "
          f"{len(r['result_expr'])} result terms")
    # pivot altitude profile
    prof = [rs(J) for J, _, _ in acts]
    maxr = max(p[0] for p in prof)
    maxs = max(p[1] for p in prof)
    print(f"pivot altitude: max r={maxr} (start {rs(T)[0]}, climb "
          f"+{maxr - rs(T)[0]}), max s={maxs}")
    # r histogram over pivots
    from collections import Counter
    hist = Counter(p[0] for p in prof)
    print("pivots per r: " + "  ".join(f"r{k}:{hist[k]}" for k in sorted(hist)))
    # where in the path the peak sits
    peak_idx = [i for i, p in enumerate(prof) if p[0] == maxr]
    print(f"peak-r pivots at action indices {peak_idx[:6]}"
          f"{'...' if len(peak_idx) > 6 else ''} of {len(acts)}")
    # simulated worker: active expr = {T:1}; each action J eliminates J via
    # sol_store[J]; track expr size + max weight of ACTIVE (>= start) terms
    kT = tkey(T)
    expr = {T: 1}
    max_terms = 0
    max_active_r = rs(T)[0]
    for i, (J, op, shift) in enumerate(acts):
        # apply substitution for J everywhere it appears (worker semantics:
        # actions ordered so J's rule exists when needed)
        if J in expr:
            co = expr.pop(J)
            for K, cK in sol[J].items():
                v = (expr.get(K, 0) + co * cK) % P
                if v:
                    expr[K] = v
                else:
                    expr.pop(K, None)
        active = [K for K in expr if tkey(K) <= kT]
        if active:
            max_active_r = max(max_active_r, max(rs(K)[0] for K in active))
        max_terms = max(max_terms, len(expr))
    n_active_end = sum(1 for K in expr if tkey(K) <= kT)
    print(f"worker-sim: peak expr size {max_terms}, peak ACTIVE-term r "
          f"{max_active_r}, active terms at end: {n_active_end} (0 = drained)")
