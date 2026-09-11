#!/usr/bin/env python
"""Estimate the worker-count savings symmetry canonicalization would give on the
real (8,5) pentagon-box reduction.

Each async worker reduced one target integral, encoded in its .sub filename:
    async_<id>_<i0>_<i1>_..._<i10>.sub   (11 indices for pentagon-box).
We collect every distinct dispatched target across the v7_fresh work dirs,
canonicalize each with Kira's pentagon-box symmetry group (sectorSymmetries +
sectorRelations), and count how many collapse to a shared representative.

Redundant workers = (distinct targets) - (distinct canonical reps).
This is the direct worker-level savings; it also quantifies the
'distinct integrals to reduce' shrink even though masters barely move (61 vs 62).
Read-only analysis."""
import sys, os, glob
from collections import Counter

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
from sailir.symmetries import parse_symmetries, SymmetryGroup, sector_of

N_INDICES = 11
N_LOOPS = 2
SYMDIR = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
WORK_DIRS = [
    os.path.join(ROOT, "results/pentagonbox_8_5_v7_fresh/work"),
    os.path.join(ROOT, "results/pentagonbox_8_5_v7_fresh_round2/work"),
]

def parse_target_from_subname(fname):
    """async_<id>_<11 ints>.sub -> tuple of 11 ints (last 11 underscore tokens)."""
    base = os.path.basename(fname)[:-len(".sub")]
    toks = base.split("_")
    ints = toks[-N_INDICES:]
    return tuple(int(x) for x in ints)

# --- collect dispatched targets ---
sub_count = 0
targets = set()
per_dir = {}
for wd in WORK_DIRS:
    subs = glob.glob(os.path.join(wd, "async_*.sub"))
    per_dir[wd] = len(subs)
    sub_count += len(subs)
    for s in subs:
        try:
            targets.add(parse_target_from_subname(s))
        except ValueError:
            print(f"  WARN could not parse: {s}")

print("=== dispatched worker targets (8,5 v7_fresh) ===")
for wd, n in per_dir.items():
    print(f"  {n:>7d} .sub  {os.path.relpath(wd, ROOT)}")
print(f"  {sub_count:>7d} .sub files total")
print(f"  {len(targets):>7d} DISTINCT target integrals\n")

# --- load symmetry group ---
syms = parse_symmetries(os.path.join(SYMDIR, "sectorSymmetries"), N_INDICES, N_LOOPS)
rels = parse_symmetries(os.path.join(SYMDIR, "sectorRelations"), N_INDICES, N_LOOPS)
group = SymmetryGroup.from_records(syms, rels)   # include_dots=True (default)
print(f"=== symmetry group: {len(syms)} sectorSymmetries + {len(rels)} sectorRelations ===\n")

# --- canonicalize every distinct target ---
reps = {}                      # rep -> list of targets mapping to it
for t in targets:
    rep = group.canonicalize(t)
    reps.setdefault(rep, []).append(t)

n_distinct = len(targets)
n_reps = len(reps)
redundant = n_distinct - n_reps
orbit_sizes = Counter(len(v) for v in reps.values())

print("=== SYMMETRY COLLAPSE ON THE REAL (8,5) WORKLOAD ===")
print(f"  distinct dispatched targets : {n_distinct}")
print(f"  distinct canonical reps     : {n_reps}")
print(f"  REDUNDANT (collapsible)     : {redundant}  ({100.0*redundant/n_distinct:.1f}% of workers)")
print(f"  effective speedup factor    : {n_distinct/n_reps:.3f}x  (targets per rep)\n")
print("  orbit-size distribution (how many reps absorb k targets):")
for k in sorted(orbit_sizes):
    print(f"    {orbit_sizes[k]:>6d} reps each absorb {k} dispatched target(s)")

# a few example collapsed orbits (size>1)
print("\n  example collapsed orbits (rep <- {targets}):")
shown = 0
for rep, members in reps.items():
    if len(members) > 1:
        print(f"    {list(rep)}  <-  {[list(m) for m in members]}")
        shown += 1
        if shown >= 8:
            break
