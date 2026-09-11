#!/usr/bin/env python
"""How many integrals does a worker's returned expression contain, and how
many of those are FRESH for the routing table (would need a solve)?

Samples banked g1023 results; checks membership against the current
production routing table (keys = every integral ever routed OR marked
unroutable).
"""
import glob
import pickle
import sys

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
G = ROOT + "/results/gr_reduce/g1023"

with open(G + "/work/sym_memo.pkl", "rb") as f:
    tab = pickle.load(f)
print(f"routing table: {len(tab)} entries", flush=True)

files = sorted(glob.glob(G + "/work/results/*.pkl"))
step = max(1, len(files) // 400)
sizes, fresh_counts = [], []
for fp in files[::step][:400]:
    try:
        with open(fp, "rb") as f:
            r = pickle.load(f)
    except Exception:
        continue
    if not r.get("success"):
        continue
    fe = r.get("final_expr", {})
    keys = [tuple(k) for k in fe]
    sizes.append(len(keys))
    fresh_counts.append(sum(1 for k in keys if k not in tab))

def stats(v):
    v = sorted(v)
    n = len(v)
    return (f"n={n} mean={sum(v)/n:.1f} med={v[n//2]} "
            f"p90={v[int(n*.9)]} max={v[-1]}")

print(f"integrals per returned expression: {stats(sizes)}", flush=True)
print(f"FRESH (need solve) per expression vs today's table: "
      f"{stats(fresh_counts)}", flush=True)
