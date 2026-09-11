#!/usr/bin/env python
"""Bracket the (8,5) symmetry worker-saving with a LOWER and an UPPER bound,
now INCLUDING cross-sector relations and accounting for combination/expansion
rows (instead of dropping them as corrected_collapse_85.py did).

LOWER bound  = clean single-rep collapse, within-sector only (what cheap
               canonicalization captures: an integral maps to ONE simpler rep).
UPPER bound  = a target is 'symmetry-touched' (Kira would not reduce it
               independently) if ANY symmetry moves it non-trivially:
                 * its sector is redundant (a cross-sector relation source), OR
                 * a within-sector matrix maps it (clean) to a different integral, OR
                 * a within-sector matrix has a COMBINATION row at one of the
                   target's nonzero slots (the numerator expands -> a real
                   symmetry relation, just not a single-rep collapse).
The true saving lies between; pinning it exactly needs the full symbolic
expansion (to check each expansion relation is actually reduction-useful)."""
import sys, os, glob
from collections import Counter, deque

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
from sailir.symmetries import parse_symmetries, sector_of

N_IDX = 11
TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
WORK_DIRS = [
    os.path.join(ROOT, "results/pentagonbox_8_5_v7_fresh/work"),
    os.path.join(ROOT, "results/pentagonbox_8_5_v7_fresh_round2/work"),
]

def parse_target(fname):
    return tuple(int(x) for x in os.path.basename(fname)[:-4].split("_")[-N_IDX:])

def parse_matrix_blocks(path):
    blocks, cur = [], []
    for ln in open(path):
        s = ln.rstrip("\n")
        if s.strip() == "":
            if cur: blocks.append(cur); cur = []
            continue
        cur.append(s)
    if cur: blocks.append(cur)
    out = []
    for b in blocks:
        if len(b) < 3 + N_IDX:
            continue
        src = int(b[1].split()[1])
        posmap = {}
        for i, row in enumerate(b[3:3 + N_IDX]):
            t = row.split()
            coeffs, const = t[:N_IDX], (t[N_IDX] if len(t) > N_IDX else "0")
            nz = [(j, c) for j, c in enumerate(coeffs) if c != "0"]
            posmap[i] = nz[0][0] if (len(nz) == 1 and nz[0][1] in ("1", "-1") and const == "0") else None
        out.append((src, posmap))
    return out

mats = parse_matrix_blocks(os.path.join(TA, "symmetries"))
by_src = {}
for src, pm in mats:
    by_src.setdefault(src, []).append(pm)
redundant = set(r.source_sector for r in
                parse_symmetries(os.path.join(TA, "sectorRelations"), N_IDX, 2))

targets = set()
for wd in WORK_DIRS:
    for s in glob.glob(os.path.join(wd, "async_*.sub")):
        targets.add(parse_target(s))
n = len(targets)

def clean_image(integral, posmap):
    new = [0] * N_IDX
    for i, a in enumerate(integral):
        if a == 0:
            continue
        j = posmap.get(i)
        if j is None or new[j] != 0:
            return None
        new[j] = a
    return tuple(new)

# ---- LOWER bound: clean within-sector single-rep collapse ----
def canon(integral):
    seen, dq, best = {integral}, deque([integral]), integral
    while dq:
        x = dq.popleft()
        for pm in by_src.get(sector_of(x), []):
            y = clean_image(x, pm)
            if y is not None and y not in seen:
                seen.add(y)
                best = min(best, y)
                dq.append(y)
    return best
reps = {}
for t in targets:
    reps.setdefault(canon(t), None)
lower_red = n - len(reps)

# ---- UPPER bound: symmetry-touched (incl. cross-sector + combinations) ----
def touched(t):
    sec = sector_of(t)
    if sec in redundant:                          # cross-sector relation maps it away
        return True
    for pm in by_src.get(sec, []):
        # combination row at a nonzero slot -> expands -> real symmetry relation
        if any(t[i] != 0 and pm.get(i) is None for i in range(N_IDX)):
            return True
        img = clean_image(t, pm)                  # clean move to a different integral
        if img is not None and img != t:
            return True
    return False
upper_red = sum(1 for t in targets if touched(t))

print(f"distinct (8,5) targets                : {n}")
print(f"broken ing-only (numerators excluded) : 386      (0.7%)")
print(f"LOWER bound  clean single-rep collapse : {lower_red:5d}    ({100*lower_red/n:.1f}%)   within-sector only")
print(f"UPPER bound  any symmetry touches it   : {upper_red:5d}    ({100*upper_red/n:.1f}%)   incl. cross-sector + combinations")
print(f"   (redundant cross-sector sectors: {sorted(redundant)})")
