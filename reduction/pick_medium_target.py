#!/usr/bin/env python
"""Pick a medium-difficulty benchmark: the target nearest the median of the
sector-senior difficulty ranking with a small nonzero FIRE answer (comparable
oracle), print it with context."""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from symmetry_route import tkey

N_DEN = 10
orc = pickle.load(open(ROOT + "/results/fire_oracle_GR.pkl", "rb"))
targets = [tuple(t) for t in orc["targets"]]
sols = {tuple(k): v for k, v in orc["solutions"].items()}
skipped = {tuple(t) for t in orc.get("skipped", [])}

by_order = sorted(targets, key=tkey)
n = len(by_order)
mid = n // 2
for off in range(n):
    for pos in (mid + off, mid - off):
        if not 0 <= pos < n:
            continue
        x = by_order[pos]
        sol = sols.get(x)
        if x in skipped or not sol:
            continue
        t = sum(1 for i in range(N_DEN) if x[i] > 0)
        r = sum(v for v in x if v > 0)
        s = sum(-v for v in x if v < 0)
        print(f"rank {pos + 1}/{n} (median={mid + 1})  t={t} r={r} s={s}")
        print(f"integral: {','.join(str(v) for v in x)}")
        print(f"FIRE answer: {len(sol)} masters")
        sys.exit(0)
