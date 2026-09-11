#!/usr/bin/env python
"""Pinpoint WHY symmetry barely collapses the (8,5) targets. Decompose the
56,767 targets through the applicability chain:
  A. in a sector that has >=1 symmetry record
  B. of those, has >=1 record that ACTUALLY APPLIES (apply_record != None)
     -- this is the test of the ing==-1 / nonzero-ISP applicability filter
  C. of those, some applicable record maps it to a DISTINCT integral (not fixed)
  D. of those, the image is ALSO a dispatched target (=> real collapse)
Also: split targets by whether they carry any nonzero ISP index (pos 8,9,10),
and report the collapse/applicability within each split."""
import sys, os, glob
from collections import Counter

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
from sailir.symmetries import parse_symmetries, SymmetryGroup, sector_of, apply_record

N_INDICES, N_LOOPS = 11, 2
ISP = (8, 9, 10)                  # pentagon-box ISP positions
SYMDIR = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
WORK_DIRS = [
    os.path.join(ROOT, "results/pentagonbox_8_5_v7_fresh/work"),
    os.path.join(ROOT, "results/pentagonbox_8_5_v7_fresh_round2/work"),
]

def parse_target(fname):
    base = os.path.basename(fname)[:-4]
    return tuple(int(x) for x in base.split("_")[-N_INDICES:])

targets = set()
for wd in WORK_DIRS:
    for s in glob.glob(os.path.join(wd, "async_*.sub")):
        targets.add(parse_target(s))

syms = parse_symmetries(os.path.join(SYMDIR, "sectorSymmetries"), N_INDICES, N_LOOPS)
rels = parse_symmetries(os.path.join(SYMDIR, "sectorRelations"), N_INDICES, N_LOOPS)
group = SymmetryGroup.from_records(syms, rels)

A = B = C = D = 0
for t in targets:
    recs = group.by_source.get(sector_of(t), [])
    if not recs:
        continue
    A += 1
    applied_imgs = []
    for r in recs:
        img = apply_record(t, r)
        if img is not None:
            applied_imgs.append(img)
    if not applied_imgs:
        continue
    B += 1
    distinct_imgs = [im for im in applied_imgs if im != t]
    if not distinct_imgs:
        continue
    C += 1
    if any(im in targets for im in distinct_imgs):
        D += 1

n = len(targets)
print(f"total distinct targets                                  : {n}")
print(f"A. in a sector with >=1 symmetry record                 : {A}  ({100*A/n:.1f}%)")
print(f"B.   ... and >=1 record APPLIES (passes ing==-1 filter)  : {B}  ({100*B/n:.1f}%)")
print(f"C.     ... and maps to a DISTINCT integral (not fixed)   : {C}  ({100*C/n:.1f}%)")
print(f"D.       ... whose image is ALSO a dispatched target     : {D}  ({100*D/n:.1f}%)  <- collapses")

# split by nonzero-ISP
def has_isp(t):
    return any(t[i] != 0 for i in ISP)
zero_isp = [t for t in targets if not has_isp(t)]
nz_isp   = [t for t in targets if has_isp(t)]
def applies_to(t):
    return any(apply_record(t, r) is not None and apply_record(t, r) != t
               for r in group.by_source.get(sector_of(t), []))
print(f"\n=== split by ISP (numerator) content ===")
print(f"targets with all-zero ISP (pos 8,9,10): {len(zero_isp)}  ({100*len(zero_isp)/n:.1f}%)")
print(f"  of these, a symmetry maps to a distinct integral: {sum(applies_to(t) for t in zero_isp)}")
print(f"targets with >=1 nonzero ISP           : {len(nz_isp)}  ({100*len(nz_isp)/n:.1f}%)")
print(f"  of these, a symmetry maps to a distinct integral: {sum(applies_to(t) for t in nz_isp)}")
