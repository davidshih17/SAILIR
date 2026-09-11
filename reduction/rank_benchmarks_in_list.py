#!/usr/bin/env python
"""Where do the 3 reduced benchmarks sit, by weight, in the full provided
target list (the FIRE oracle's ~2100 targets)? Ranks by the sector-senior
total order and by plain (t, r, s)."""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from symmetry_route import tkey
from collections import Counter

N_DEN = 10
orc = pickle.load(open(ROOT + "/results/fire_oracle_GR.pkl", "rb"))
targets = [tuple(t) for t in orc["targets"]]
oracle = {tuple(k): v for k, v in orc["solutions"].items()}

TAGS = {
    (1, 0, 1, -1, 1, 1, 1, 0, 1, 1, -1, 0, 0, 0, 0): "g885",
    (1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0): "g127",
    (1, 0, 1, 1, 1, 1, 1, 0, 1, 1, -1, 0, 0, 0, -1): "g893",
}


def trs(x):
    t = sum(1 for i in range(N_DEN) if x[i] > 0)
    r = sum(v for v in x if v > 0)
    s = sum(-v for v in x if v < 0)
    return t, r, s


print(f"full list: {len(targets)} targets")
c = Counter(trs(x) for x in targets)
print("\n(t=dens, r=dots+dens, s=numerators): count   [top rows by t,r,s]")
for k in sorted(c, reverse=True)[:14]:
    print(f"  t={k[0]:2d} r={k[1]:2d} s={k[2]}: {c[k]:4d}")
print("  ...")

# rank by sector-senior order: rank 1 = HARDEST (highest in the order)
by_order = sorted(targets, key=tkey)     # smallest tkey = highest/hardest
n = len(by_order)
print(f"\nbenchmark positions (rank 1 = hardest of {n}, sector-senior order):")
for x, tag in TAGS.items():
    pos = by_order.index(x) + 1
    t, r, s = trs(x)
    nm = len(oracle[x])
    print(f"  {tag}: rank {pos:4d} / {n}  (top {100 * pos / n:5.1f}%)   "
          f"t={t} r={r} s={s}   FIRE answer: {nm} masters")

# context: the very hardest entries
print("\nhardest 5 in the list:")
for x in by_order[:5]:
    t, r, s = trs(x)
    print(f"  t={t} r={r} s={s}  {list(x)}")
