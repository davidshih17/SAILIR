#!/usr/bin/env python
"""Predict whether CHEAP canonicalization (single-rep collapse) can capture a
family's symmetry, by classifying each record's loop_subst as:
  PURE-PERM : every rhs is a bare +/- loop momentum (k_i) -> numerators map
              cleanly to a single integral (cheap canonicalization works).
  SHIFT     : some rhs contains an external momentum -> numerators EXPAND into a
              sum (needs symmetry-as-relation, i.e. option A).
High pure-perm fraction => cheap canon captures most of the symmetry."""
import sys, os
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
from sailir.symmetries import parse_symmetries

def loops_only(rhs, loop_names):
    """True if rhs is a bare +/- single loop momentum (no external momenta)."""
    body = rhs.lstrip("+-")
    return body in loop_names

CASES = [
    ("pentagonbox TA", "results/kira_reduce_161/sectormappings/TA", 11, 2, {"k1", "k2"}),
    ("gravity GR",     "topology_input/gravity3L/kira_validate/sectormappings/GR", 15, 3,
     {"k1", "k2", "k3"}),
]

for name, sub, n_idx, n_loops, loops in CASES:
    d = os.path.join(ROOT, sub)
    print(f"=== {name} ===")
    for fn in ("sectorSymmetries", "sectorRelations"):
        recs = parse_symmetries(os.path.join(d, fn), n_idx, n_loops)
        pure = sum(1 for r in recs if all(loops_only(rhs, loops) for _, rhs in r.loop_substs))
        shift = len(recs) - pure
        frac = 100 * pure / len(recs) if recs else 0
        print(f"  {fn:18s}: {len(recs):4d} records | PURE-PERM {pure:4d} ({frac:4.1f}%) | SHIFT {shift:4d}")
    print()
