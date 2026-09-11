#!/usr/bin/env python
"""Intermediate agreement check: does the NEW (post-strip) design1 run agree with the
PREVIOUS (pre-strip) design1 run, per integral, over the integrals both have reduced
so far? Per-worker result files are keyed by integral (async_<integral>.pkl) and hold
final_expr (the reduction). For every integral present in BOTH work/results dirs, the
reductions must be identical."""
import os, sys, pickle
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
ROOT = os.path.join(BASE, "results/ab_symmetry")

def load_results(tag_arm):
    d = os.path.join(ROOT, tag_arm, "work", "results")
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        if not fn.endswith(".pkl"):
            continue
        try:
            r = pickle.load(open(os.path.join(d, fn), "rb"))
        except Exception:
            continue
        key = tuple(r["original_integral"])
        fe = {tuple(i): c for i, c in r["final_expr"].items() if c != 0}
        out[key] = fe
    return out

for label, new, old in [("m1", "m1_symcheck/design1", "m1_6prop/design1"),
                        ("m3", "m3_symcheck/design1", "m3_5prop_deg3/design1")]:
    N = load_results(new); O = load_results(old)
    overlap = set(N) & set(O)
    match = sum(1 for k in overlap if N[k] == O[k])
    differ = [k for k in overlap if N[k] != O[k]]
    print(f"[{label}] new={len(N)} old={len(O)} overlap={len(overlap)}  "
          f"MATCH={match}  DIFFER={len(differ)}  "
          f"{'ALL AGREE ✓' if not differ else '*** MISMATCH ***'}")
    for k in differ[:5]:
        print(f"    DIFFER I{list(k)}:\n      new={dict(list(N[k].items())[:4])}\n      old={dict(list(O[k].items())[:4])}")
