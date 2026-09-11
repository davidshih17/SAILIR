#!/usr/bin/env python
"""Measure the (8,5) symmetry WIN count under the criterion:
   a target I is a win if SOME symmetry maps it to an expansion whose every term
   (other than I) is STRICTLY SIMPLER than I (lower t,r,s,d,sector) OR already in
   the worker list. Includes within-sector (raw matrices) AND cross-sector
   (derived via the momentum-algebra engine), with full numerator expansion."""
import sys, os, glob, time
from collections import Counter

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir.symmetries import parse_symmetries, sector_of, default_order_key
from symmetry_engine import (transform_support, raw_matrix_to_Mc, derived_to_Mc,
                             ing_to_Mc, is_placeholder, N)

TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
WORK_DIRS = [os.path.join(ROOT, "results/pentagonbox_8_5_v7_fresh/work"),
             os.path.join(ROOT, "results/pentagonbox_8_5_v7_fresh_round2/work")]

# ---- self-check: a pure k1<->k2 swap must give a SINGLE relabeled image ----
rels = parse_symmetries(os.path.join(TA, "sectorRelations"), N, 2)
swap = next(r for r in rels if all(rhs in ("k1", "k2") for _, rhs in r.loop_substs))
M, cnz = derived_to_Mc(list(swap.loop_substs))
# pick an integral in the swap's source sector with a numerator
src = swap.source_sector
probe = tuple((2 if (src >> i) & 1 else (-1 if i == 8 else 0)) for i in range(N))
supp = transform_support(probe, M, cnz)
print(f"[self-check] pure swap {swap.source_sector}->{swap.target_sector} loop_subst={swap.loop_substs}")
print(f"  probe {probe} -> support {supp}")
print(f"  single image? {supp is not None and supp != 'LARGE' and len(supp)==1}\n")

# ---- build symmetry tables ----
# within-sector: parse raw matrices
blocks, cur = [], []
for ln in open(os.path.join(TA, "symmetries")):
    s = ln.rstrip("\n")
    if s.strip() == "":
        if cur: blocks.append(cur); cur = []
        continue
    cur.append(s)
if cur: blocks.append(cur)
within = {}   # source sector -> list of (M, c_nonzero)
for b in blocks:
    src_sec = int(b[1].split()[1])
    M, cnz = raw_matrix_to_Mc(b[3:3 + N])
    within.setdefault(src_sec, []).append((M, cnz))

# cross-sector: derive via engine
t0 = time.time()
cross = {}
n_ph = 0
for r in rels:
    if is_placeholder(r.loop_substs):
        M, cnz = ing_to_Mc(r.ing); n_ph += 1
    else:
        M, cnz = derived_to_Mc(list(r.loop_substs))
    cross.setdefault(r.source_sector, []).append((M, cnz))
print(f"derived {len(rels)-n_ph} cross-sector transforms ({n_ph} placeholder->ing) in {time.time()-t0:.1f}s")
print(f"within-sector sectors: {len(within)}, cross-sector source sectors: {len(cross)}\n")

# ---- load targets ----
targets = set()
for wd in WORK_DIRS:
    for s in glob.glob(os.path.join(wd, "async_*.sub")):
        targets.add(tuple(int(x) for x in os.path.basename(s)[:-4].split("_")[-N:]))
n = len(targets)
key = default_order_key

def classify(I):
    """Return 'strict' (maps to all-strictly-simpler), 'inlist' (needs a same-level
    in-list term), 'large' (some applicable expansion too big to enumerate), or None."""
    kI = key(I)
    secI = sector_of(I)
    saw_large = False
    inlist_only = False
    for table in (within, cross):
        for (M, cnz) in table.get(secI, []):
            supp = transform_support(I, M, cnz)
            if supp is None:
                continue
            if supp == "LARGE":
                saw_large = True
                continue
            others = supp - {I}
            if not others:
                continue
            if all(key(t) < kI for t in others):
                return "strict"                       # genuine reduction, no over-count
            if all((t in targets) or (key(t) < kI) for t in others):
                inlist_only = True
    if inlist_only:
        return "inlist"
    return "large" if saw_large else None

t0 = time.time()
cnt = Counter()
for idx, I in enumerate(targets):
    cnt[classify(I)] += 1
    if (idx + 1) % 20000 == 0:
        print(f"  ... {idx+1}/{n} ({time.time()-t0:.0f}s)")

strict, inlist, large = cnt["strict"], cnt["inlist"], cnt["large"]
wins = strict + inlist
print(f"\n=== (8,5) symmetry WINS, full validated transform ===")
print(f"  distinct targets               : {n}")
print(f"  strictly-simpler (clean win)   : {strict:6d}  ({100*strict/n:.1f}%)")
print(f"  + same-level in-list           : {inlist:6d}  ({100*inlist/n:.1f}%)")
print(f"  = TOTAL WINS (in-list-or-simpler): {wins:6d}  ({100*wins/n:.1f}%)")
print(f"  large-expansion (skipped, under-count): {large}  ({100*large/n:.1f}%)")
print(f"  (broken ing-only 0.7% | clean single-rep 3.4% | touched upper 42%)")
