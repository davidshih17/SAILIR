#!/usr/bin/env python
"""Corrected (8,5) collapse measurement using the FULL within-sector transform
matrices (raw 'symmetries' file), NOT the denominator-only ing.

For each within-sector symmetry, parse its 11x12 matrix into a per-position map:
  row i is CLEAN if exactly one of cols 0..10 is +/-1 and the constant col 11 is 0
          -> propagator i maps to a single propagator (relabeling, incl. ISP slots)
  else COMBINATION (numerator there would expand; not a cheap relabeling).

A target collapses cleanly if ALL its nonzero slots have CLEAN rows; then we
relabel (ignoring sign, which only affects coefficients not the collapse count)
and take the orbit minimum. Compare to the broken ing-only 0.7%.

Scope: WITHIN-sector only (the 111 matrices). Cross-sector relations (21) are not
in the raw matrix file and need the loop_subst derivation -- handled separately."""
import sys, os, glob
from collections import Counter, deque

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
from sailir.symmetries import sector_of

N_IDX = 11
TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
WORK_DIRS = [
    os.path.join(ROOT, "results/pentagonbox_8_5_v7_fresh/work"),
    os.path.join(ROOT, "results/pentagonbox_8_5_v7_fresh_round2/work"),
]

def parse_target(fname):
    return tuple(int(x) for x in os.path.basename(fname)[:-4].split("_")[-N_IDX:])

# --- parse raw symmetry matrices into per-position clean/combo maps ---
def parse_matrix_blocks(path):
    blocks, cur = [], []
    for ln in open(path):
        s = ln.rstrip("\n")
        if s.strip() == "":
            if cur: blocks.append(cur); cur = []
            continue
        cur.append(s)
    if cur: blocks.append(cur)
    out = []   # (src_sector, posmap)  posmap[i] = j (clean perm) or None (combo)
    for b in blocks:
        if len(b) < 3 + N_IDX:
            continue
        src = int(b[1].split()[1])
        rows = b[3:3 + N_IDX]
        posmap = {}
        for i, row in enumerate(rows):
            t = row.split()
            coeffs = t[:N_IDX]            # cols 0..10
            const = t[N_IDX] if len(t) > N_IDX else "0"
            nz = [(j, c) for j, c in enumerate(coeffs) if c not in ("0",)]
            if len(nz) == 1 and nz[0][1] in ("1", "-1") and const == "0":
                posmap[i] = nz[0][0]     # clean: prop i -> prop nz[0][0]
            else:
                posmap[i] = None         # combination
        out.append((src, posmap))
    return out

mats = parse_matrix_blocks(os.path.join(TA, "symmetries"))
by_src = {}
for src, pm in mats:
    by_src.setdefault(src, []).append(pm)
print(f"parsed {len(mats)} within-sector matrices over {len(by_src)} sectors")

def clean_image(integral, posmap):
    """Relabel via posmap if ALL nonzero slots are clean; else None."""
    new = [0] * N_IDX
    for i, a in enumerate(integral):
        if a == 0:
            continue
        j = posmap.get(i)
        if j is None:                    # nonzero slot hits a combination row
            return None
        if new[j] != 0:                  # collision -> not a clean permutation here
            return None
        new[j] = a
    return tuple(new)

def canon(integral):
    """Orbit minimum under clean within-sector relabelings (lex order)."""
    seen = {integral}
    dq = deque([integral])
    best = integral
    while dq:
        x = dq.popleft()
        for pm in by_src.get(sector_of(x), []):
            y = clean_image(x, pm)
            if y is not None and y not in seen:
                seen.add(y)
                if y < best:
                    best = y
                dq.append(y)
    return best

# --- load targets, measure collapse ---
targets = set()
for wd in WORK_DIRS:
    for s in glob.glob(os.path.join(wd, "async_*.sub")):
        targets.add(parse_target(s))

reps = {}
mappable = 0
for t in targets:
    r = canon(t)
    if r != t:
        mappable += 1
    reps.setdefault(r, []).append(t)

n = len(targets)
red = n - len(reps)
print(f"\n=== CORRECTED within-sector collapse on (8,5) ===")
print(f"  distinct targets           : {n}")
print(f"  distinct canonical reps    : {len(reps)}")
print(f"  REDUNDANT (collapsible)    : {red}  ({100*red/n:.1f}% of workers)")
print(f"  (broken ing-only baseline  : 386, 0.7%)")
osz = Counter(len(v) for v in reps.values())
print(f"  orbit-size distribution    : {dict(sorted(osz.items()))}")

# how many targets have a numerator (nonzero ISP) AND collapsed -- the recovered ones
ISP = (8, 9, 10)
recovered = sum(1 for r, ms in reps.items() for m in ms
                if len(ms) > 1 and any(m[i] != 0 for i in ISP))
print(f"  targets with a numerator that now participate in a collapse: {recovered}")
