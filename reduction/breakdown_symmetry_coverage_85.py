#!/usr/bin/env python
"""Explain the (8,5) collapse rate: break the dispatched targets down by sector,
and cross with which sectors actually carry symmetry records. Tests the
hypothesis that the high-weight targets live in asymmetric top sectors (esp.
255, the full 8-propagator corner, which has NO symmetry record)."""
import sys, os, glob
from collections import Counter

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
from sailir.symmetries import parse_symmetries, SymmetryGroup, sector_of

N_INDICES, N_LOOPS = 11, 2
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
sym_sectors = set(group.by_source.keys())          # sectors with >=1 record

# targets per sector
per_sector = Counter(sector_of(t) for t in targets)
n_total = len(targets)
in_sym = sum(c for s, c in per_sector.items() if s in sym_sectors)

print(f"=== {n_total} distinct targets across {len(per_sector)} sectors ===")
print(f"sectors carrying >=1 symmetry record: {len(sym_sectors)}")
print(f"targets in a symmetric sector : {in_sym}  ({100.0*in_sym/n_total:.1f}%)")
print(f"targets in an ASYMMETRIC sector: {n_total-in_sym}  ({100.0*(n_total-in_sym)/n_total:.1f}%)  <- cannot collapse\n")

print("=== top 15 sectors by target count (sym? = has symmetry records) ===")
for sec, cnt in per_sector.most_common(15):
    nrec = len(group.by_source.get(sec, []))
    tag = f"SYM({nrec} recs)" if sec in sym_sectors else "asym"
    print(f"  sector {sec:>3d}: {cnt:>6d} targets   [{tag}]")

# collapse rate RESTRICTED to targets in symmetric sectors
sym_targets = [t for t in targets if sector_of(t) in sym_sectors]
reps = {}
for t in sym_targets:
    reps.setdefault(group.canonicalize(t), []).append(t)
red = len(sym_targets) - len(reps)
print(f"\n=== collapse among the {len(sym_targets)} targets that ARE in symmetric sectors ===")
print(f"  distinct: {len(sym_targets)}  reps: {len(reps)}  redundant: {red} "
      f"({100.0*red/max(1,len(sym_targets)):.1f}% of symmetric-sector targets)")
