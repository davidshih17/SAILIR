#!/usr/bin/env python
"""Concretely verify the conceptual claim on the pentagon-box (TA family):

(A1) Every Kira symmetry's `ing` is a PERMUTATION of the SOURCE sector's present
     propagators onto the TARGET sector's present propagators (the denominator
     part of the symmetry IS a propagator permutation).
(A2) The raw `symmetries` matrices confirm: rows of PRESENT propagators are
     single-entry +/-1 with ZERO constant (a permutation), while the ISP rows
     (absent-denominator + ISP slots) are COMBINATIONS with kinematic constants
     (the numerator expansion).
(B)  Cross-sector relations partition sectors into ORBITS -> print orbits,
     representatives, and the eliminated (non-representative) sectors; check 161
     (the symmetry-redundant 62nd master) is eliminated.

Output is unbuffered; run in background, read the log.
"""
import os, sys
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir.symmetries import parse_symmetries
from symmetry_engine import raw_matrix_to_Mc      # uses module N=11 internally

TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
NIDX, NPROP = 11, 8
def present(sec): return {i for i in range(NPROP) if (sec >> i) & 1}

print("=" * 72, flush=True)
print("(A1) ing == permutation of present propagators (source -> target)?", flush=True)
for fname in ("sectorSymmetries", "sectorRelations"):
    recs = parse_symmetries(os.path.join(TA, fname), NIDX, 2)
    npass = nfail = 0; fails = []
    for r in recs:
        pS, pT = present(r.source_sector), present(r.target_sector)
        mapped = {g for g in range(NIDX) if r.ing[g] != -1}
        images = {r.ing[g] for g in range(NIDX) if r.ing[g] != -1}
        ok = (mapped == pS) and (images == pT) and (len(mapped) == len(images))
        if ok: npass += 1
        else:
            nfail += 1
            if len(fails) < 6:
                fails.append((r.source_sector, r.target_sector,
                              sorted(mapped), sorted(pS), sorted(images), sorted(pT)))
    print(f"  {fname:18s}: {len(recs):4d} records   PASS={npass}   FAIL={nfail}", flush=True)
    for f in fails:
        print(f"     FAIL src={f[0]} tgt={f[1]} mapped={f[2]} presS={f[3]} "
              f"images={f[4]} presT={f[5]}", flush=True)

print("=" * 72, flush=True)
print("(A2) raw matrices: present-prop ROWS = single +/-1 permutation (0 const);", flush=True)
print("     ISP rows = combinations (numerator expansion)?", flush=True)
blocks, cur = [], []
for ln in open(os.path.join(TA, "symmetries")):
    s = ln.rstrip("\n")
    if s.strip() == "":
        if cur: blocks.append(cur); cur = []
    else:
        cur.append(s)
if cur: blocks.append(cur)

nblk = nperm = nispcombo = 0; bad = []
for b in blocks:
    if len(b) < 3 + NIDX: continue
    src = int(b[1].split()[1]); tgt = int(b[2].split()[1])
    M, c = raw_matrix_to_Mc(b[3:3 + NIDX])
    pS, pT = present(src), present(tgt)
    nblk += 1
    perm_ok = True; images = []
    for i in pS:                                   # present-propagator rows
        row = M.get(i, {})
        if len(row) == 1 and not c.get(i, False):
            (j, v), = row.items()
            if j in pT and abs(v) == 1:
                images.append(j); continue
        perm_ok = False; break
    if perm_ok and sorted(images) == sorted(pT): nperm += 1
    elif len(bad) < 6: bad.append((src, tgt, sorted(pS), sorted(pT)))
    isp_rows = [i for i in range(NIDX) if i not in pS]   # absent-denom + ISP slots
    if any(len(M.get(i, {})) > 1 or c.get(i, False) for i in isp_rows): nispcombo += 1
print(f"  blocks={nblk}   present-prop-rows-form-a-permutation={nperm}", flush=True)
for x in bad:
    print(f"     NON-PERM block: src={x[0]} tgt={x[1]} presS={x[2]} presT={x[3]}", flush=True)
print(f"  blocks with >=1 ISP row that is a COMBINATION (numerator expansion): "
      f"{nispcombo}/{nblk}", flush=True)

print("=" * 72, flush=True)
print("(B) sector ORBITS from cross-sector relations", flush=True)
rels = parse_symmetries(os.path.join(TA, "sectorRelations"), NIDX, 2)
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[max(ra, rb)] = min(ra, rb)   # representative = MIN sector id
sectors = set()
for r in rels:
    sectors.add(r.source_sector); sectors.add(r.target_sector)
    if r.source_sector != r.target_sector:
        union(r.source_sector, r.target_sector)
orbits = {}
for s in sectors:
    orbits.setdefault(find(s), set()).add(s)
multi = {rep: m for rep, m in orbits.items() if len(m) > 1}
elim = sorted(s for rep, m in orbits.items() for s in m if s != rep)
print(f"  sectors in sectorRelations: {len(sectors)}", flush=True)
print(f"  orbits: {len(orbits)}   (non-trivial size>1: {len(multi)})", flush=True)
print(f"  eliminated (non-representative) sectors: {len(elim)} -> {elim}", flush=True)
for rep, m in sorted(multi.items()):
    print(f"     rep={rep:3d}  members={sorted(m)}  eliminate={sorted(x for x in m if x!=rep)}", flush=True)
print(f"  sector 161 eliminated? "
      f"{('YES (rep=%d)' % find(161)) if 161 in elim else 'no'}", flush=True)
print("DONE", flush=True)
