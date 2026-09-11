#!/usr/bin/env python
"""Compare the *_deltafix pentagonbox arms against the *_primefix references:
final master expression must be IDENTICAL (same masters, same mod-p
coefficients). Also reports worker counts as a secondary signal (path
identity)."""
import os, glob, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
ARMS = [("gate_small_deltafix", "gate_small_primefix"),
        ("m1_deltafix", "m1_primefix"),
        ("m2_deltafix", "m2_primefix"),
        ("m3_deltafix", "m3_primefix")]

all_ok = True
for new, ref in ARMS:
    nd = os.path.join(ROOT, "results/ab_symmetry", new, "design1")
    rd = os.path.join(ROOT, "results/ab_symmetry", ref, "design1")
    np_, rp = os.path.join(nd, "reduction.pkl"), os.path.join(rd, "reduction.pkl")
    if not os.path.exists(np_):
        print(f"{new}: not finished yet")
        all_ok = False
        continue
    a = pickle.load(open(np_, "rb"))
    b = pickle.load(open(rp, "rb"))
    ea = {tuple(k): v % 1009 for k, v in a["final_expr"].items() if v % 1009}
    eb = {tuple(k): v % 1009 for k, v in b["final_expr"].items() if v % 1009}
    wn = len(glob.glob(os.path.join(nd, "work/results/*.pkl")))
    wr = len(glob.glob(os.path.join(rd, "work/results/*.pkl")))
    same = (ea == eb)
    all_ok &= same
    print(f"{new} vs {ref}: final_expr "
          f"{'IDENTICAL' if same else 'DIFFERS'} "
          f"({len(ea)} vs {len(eb)} terms); workers {wn} vs {wr}")
    if not same:
        onlya = set(ea) - set(eb); onlyb = set(eb) - set(ea)
        diff = {k for k in set(ea) & set(eb) if ea[k] != eb[k]}
        print(f"   only-new {len(onlya)}, only-ref {len(onlyb)}, "
              f"coeff-diff {len(diff)}")
print("ALL PASS — nothing changed" if all_ok else "NOT YET / MISMATCH")
