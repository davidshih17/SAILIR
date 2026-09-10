#!/usr/bin/env python
"""Rebuild the GR sector-orbit structure from the VERIFIED transform store
(gr_transforms.pkl) and compare against the ing-based canonical_sectors_GR.pkl
(which the canon10x training data restriction used).

The ing-based build trusted Kira's permutation arrays at sector level; the
pentagonbox lesson is that 'fixed' ing entries can be fake. Here every edge is a
verified corner->corner single-term image under an exact value identity, so this
is the ground-truth orbit structure. Differences would mean the training-data
canonical set was wrong (report loudly; decide then)."""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
os.environ.setdefault('SAILIR_TOPOLOGY', 'gravity3L')
import canonicalize_GR as CG

N_DEN, N = CG.N_DEN, CG.N


def corner(mask):
    return tuple(1 if mask >> i & 1 else 0 for i in range(N_DEN)) + (0,) * (N - N_DEN)


def sec(t):
    return sum(1 << i for i in range(N_DEN) if t[i] > 0)


# The BASE order (no sector-rank prefix) -- these scripts BUILD the sector
# ranking, so using the rank-prefixed tkey here would be circular. One
# definition: reduction/total_order.py.
from total_order import base_key as tkey  # noqa: E402


# verified corner graph
edges = {}
for m in range(1, 1 << N_DEN):
    out = set()
    for (M, c) in CG._transforms(corner(m)):
        img = CG.image_unsigned(corner(m), M, c)
        if img is None or len(img) != 1:
            continue
        (J, co), = img.items()
        out.add(sec(J))
    edges[m] = out

rep_of = {}
seen = set()
orbits = []
for S in range(1, 1 << N_DEN):
    if S in seen:
        continue
    orb = {S}; frontier = [S]
    while frontier:
        T = frontier.pop()
        for U in edges[T]:
            if U not in orb:
                orb.add(U); frontier.append(U)
    rep = sec(max((corner(T) for T in orb), key=tkey))
    for T in orb:
        rep_of[T] = rep
    seen |= orb
    orbits.append(sorted(orb))

canonical = sorted(set(rep_of.values()))
print(f"verified-map orbits: {len(orbits)}, canonical sectors: {len(canonical)}")

with open(os.path.join(ROOT, "results/canonical_sectors_GR.pkl"), "rb") as f:
    old = pickle.load(f)
old_rep, old_canon = old["rep_of"], set(old["canonical"])

same_rep = sum(1 for S in range(1, 1 << N_DEN) if rep_of[S] == old_rep[S])
print(f"rep_of agreement: {same_rep}/{(1 << N_DEN) - 1}")
new_only = sorted(set(canonical) - old_canon)
old_only = sorted(old_canon - set(canonical))
print(f"canonical sets: ing-based {len(old_canon)}, verified {len(canonical)}")
print(f"  canonical only in verified build (LESS merging): {len(new_only)} {new_only[:15]}")
print(f"  canonical only in ing-based build (MORE merging): {len(old_only)} {old_only[:15]}")
if same_rep == (1 << N_DEN) - 1:
    print("ALL PASS — ing-based canonical sectors confirmed by verified maps")
else:
    print("MISMATCH — training-data canonical set differs from verified orbits")
