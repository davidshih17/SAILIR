#!/usr/bin/env python
"""Test the hypothesis that ing=-1 at an ISP position does NOT mean 'integral
must be 0 there' (our apply_record rule) but merely 'not a denominator -- see the
full transform'. Compare the distilled ing (sectorSymmetries) against the raw
linear transform (symmetries matrices) for the same sector, focusing on the ISP
rows (pentagon-box positions 8,9,10)."""
import sys, os
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
from sailir.symmetries import parse_symmetries

TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
N_IDX = 11
ISP = (8, 9, 10)

# ing from sectorSymmetries, for source sector 53
syms = parse_symmetries(os.path.join(TA, "sectorSymmetries"), N_IDX, 2)
recs53 = [r for r in syms if r.source_sector == 53]
print(f"sectorSymmetries records with source 53: {len(recs53)}")
for r in recs53:
    print(f"  ing={r.ing}  loop_substs={r.loop_substs}  sign={r.sign}")
print(f"  -> apply_record would REQUIRE integral[8]=integral[9]=integral[10]=0 (ing=-1 there)\n")

# raw symmetries: parse blocks (flag / src / tgt / 11 matrix rows / blank)
blocks = []
cur = []
for ln in open(os.path.join(TA, "symmetries")):
    s = ln.rstrip("\n")
    if s.strip() == "":
        if cur: blocks.append(cur); cur = []
        continue
    cur.append(s)
if cur: blocks.append(cur)

# find blocks whose target sector (line[2]) == 53
b53 = [b for b in blocks if int(b[2].split()[1]) == 53]
print(f"raw 'symmetries' matrix blocks with sector 53: {len(b53)}")
b = b53[0]
rows = b[3:3 + N_IDX]
print(f"  first block, full {N_IDX}x12 transform matrix (row i = how propagator i maps):")
for i, row in enumerate(rows):
    tag = "  <-- ISP" if i in ISP else ""
    print(f"    pos {i:>2d}: {row}{tag}")
print()
print("  KEY: if an ISP row is a clean unit vector (e.g. pos8 -> '...1...' at col8),")
print("       the symmetry maps that numerator CLEANLY -- ing's -1 there is 'not a")
print("       denominator', NOT 'must be zero'. If the ISP row is a combination")
print("       (multiple coeffs / a kinematic constant in the last column), the")
print("       numerator EXPANDS into several integrals.")
