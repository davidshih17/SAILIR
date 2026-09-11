#!/usr/bin/env python
"""Decisive canon-correctness gate (spec section 7.2):
   canonicalizing Kira's 62 no-sym masters must collapse them to the 61 sym
   masters (set-equal). If it does NOT, the existing canon undercounts and the
   measured (8,5) collapse rate is an artifact rather than a finding."""
import sys, os, re

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
from sailir.symmetries import parse_symmetries, SymmetryGroup

N_INDICES, N_LOOPS = 11, 2
SYMDIR = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")

def load_masters(path):
    out = []
    for ln in open(path):
        m = re.match(r"\s*\w+\[([^\]]+)\]", ln)
        if m:
            out.append(tuple(int(x) for x in m.group(1).split(",")))
    return out

sym61   = load_masters(os.path.join(ROOT, "topology_input/pentagonbox/masters"))
nosym62 = load_masters(os.path.join(ROOT, "topology_input/pentagonbox_nosym/masters"))

syms = parse_symmetries(os.path.join(SYMDIR, "sectorSymmetries"), N_INDICES, N_LOOPS)
rels = parse_symmetries(os.path.join(SYMDIR, "sectorRelations"), N_INDICES, N_LOOPS)
group = SymmetryGroup.from_records(syms, rels)

canon_reps = {}
for m in nosym62:
    canon_reps.setdefault(group.canonicalize(m), []).append(m)

print(f"sym masters (truth)      : {len(sym61)}")
print(f"no-sym masters (input)   : {len(nosym62)}")
print(f"canon(no-sym) distinct   : {len(canon_reps)}   (want {len(sym61)})")
print()

set_sym = set(sym61)
set_rep = set(canon_reps.keys())
print(f"reps == sym basis (set)  : {set_rep == set_sym}")
extra = set_rep - set_sym
missing = set_sym - set_rep
if extra:
    print(f"  reps NOT in sym basis ({len(extra)}): {[list(x) for x in list(extra)[:6]]}")
if missing:
    print(f"  sym masters NOT hit  ({len(missing)}): {[list(x) for x in list(missing)[:6]]}")

# which no-sym master(s) actually collapsed (the 62->61 pair)
collapsed = {r: ms for r, ms in canon_reps.items() if len(ms) > 1}
print(f"\ncollapsed orbits (>1 no-sym master share a rep): {len(collapsed)}")
for r, ms in collapsed.items():
    print(f"  rep {list(r)}  <-  {[list(m) for m in ms]}")

verdict = "PASS" if (len(canon_reps) == len(sym61) and set_rep == set_sym) else "FAIL"
print(f"\nGATE: {verdict}")
