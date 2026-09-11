#!/usr/bin/env python
"""Test the claim: a symmetry relation never lowers the LEADING weight -- it
exchanges I for another same-(t,r,s,d) integral + strictly-lower subweight terms.
For every (8,5) symmetry application, compare the max image weight to the source
weight (t,r,s,d only; sector/tuple ignored)."""
import sys, os, glob
from collections import Counter

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir.symmetries import parse_symmetries, sector_of
from symmetry_engine import (transform_support, raw_matrix_to_Mc, derived_to_Mc,
                             ing_to_Mc, is_placeholder, N)

TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
WORK = [os.path.join(ROOT, "results/pentagonbox_8_5_v7_fresh/work"),
        os.path.join(ROOT, "results/pentagonbox_8_5_v7_fresh_round2/work")]

def weight(a):
    pos = [x for x in a if x > 0]; neg = [x for x in a if x <= 0]
    t = len(pos); r = sum(pos); s = -sum(neg); d = sum(x - 1 for x in pos if x > 1)
    return (t, r, s, d)

# build symmetry tables
blocks, cur = [], []
for ln in open(os.path.join(TA, "symmetries")):
    s = ln.rstrip("\n")
    if not s.strip():
        if cur: blocks.append(cur); cur = []
        continue
    cur.append(s)
if cur: blocks.append(cur)
within = {}
for b in blocks:
    within.setdefault(int(b[1].split()[1]), []).append(raw_matrix_to_Mc(b[3:3+N]))
cross = {}
for r in parse_symmetries(os.path.join(TA, "sectorRelations"), N, 2):
    Mc = ing_to_Mc(r.ing) if is_placeholder(r.loop_substs) else derived_to_Mc(list(r.loop_substs))
    cross.setdefault(r.source_sector, []).append(Mc)

targets = set()
for wd in WORK:
    for f in glob.glob(os.path.join(wd, "async_*.sub")):
        targets.add(tuple(int(x) for x in os.path.basename(f)[:-4].split("_")[-N:]))

cmp = Counter()       # 'same' | 'lower' | 'higher' for the max image weight vs source
n_apply = 0
for I in targets:
    wI = weight(I)
    for table in (within, cross):
        for (M, cnz) in table.get(sector_of(I), []):
            supp = transform_support(I, M, cnz)
            if supp is None or supp == "LARGE":
                continue
            others = supp - {I}
            if not others:
                continue
            n_apply += 1
            wmax = max(weight(t) for t in others)
            cmp['same' if wmax == wI else ('higher' if wmax > wI else 'lower')] += 1

print(f"symmetry applications with a non-trivial image: {n_apply}")
print(f"  max image weight == source weight (EXCHANGE)        : {cmp['same']}  ({100*cmp['same']/max(1,n_apply):.1f}%)")
print(f"  max image weight  < source weight (genuine LOWERING): {cmp['lower']} ({100*cmp['lower']/max(1,n_apply):.1f}%)")
print(f"  max image weight  > source weight (raises weight!)  : {cmp['higher']}")
