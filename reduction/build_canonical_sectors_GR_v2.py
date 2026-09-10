#!/usr/bin/env python
"""GR canonical sectors v2 — CLEAN-DEN convention (supersedes the ing-based
results/canonical_sectors_GR.pkl).

The ing-based v1 orbits were sign-blind: 6 sectors (305, 309, 369, 721, 825,
881) merge into their orbits only through FLIP maps (a linear denominator
mapping with coefficient -1), which are NOT value identities (empirically
locked by analysis/test_merge_signs_vs_fire.py — see canonicalize_GR.py).
Under clean-only edges those merges dissolve: 298 canonical sectors, and 16
sectors change representative.

Edges here are corner->corner single-term coefficient-1 images under the
numerically-verified transform store, with the clean-den image function —
every edge is a true value identity. Rep of each orbit = the _target_key-MAX
corner (same convention as v1/pentagonbox).

NOTE (training-data gap): the canon10x dataset was restricted to the v1 292
set; the 6 newly-canonical sectors have no training scrambles. The router will
lawfully emit them as survivors — the model must generalize there. Assess in
the benchmark; a data top-off is a later decision.

Outputs: results/canonical_sectors_GR_v2.pkl / .txt
"""
import os, sys, pickle
os.environ.setdefault('SAILIR_TOPOLOGY', 'gravity3L')
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
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


edges = {}
for m in range(1, 1 << N_DEN):
    out = set()
    for (M, c) in CG._transforms(corner(m)):
        img = CG.image_unsigned(corner(m), M, c)
        if img is None or len(img) != 1:
            continue
        (J, co), = img.items()
        if co % CG.P == 1:
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
print(f"clean-den orbits: {len(orbits)}, canonical sectors: {len(canonical)} "
      f"(zoom {1023/len(canonical):.2f}x)")

old = pickle.load(open(os.path.join(ROOT, "results/canonical_sectors_GR.pkl"), "rb"))
newly = sorted(set(canonical) - set(old["canonical"]))
print(f"newly canonical vs v1 (no training scrambles!): {newly}")
changed = [S for S in range(1, 1 << N_DEN) if rep_of[S] != old["rep_of"][S]]
print(f"sectors with changed rep: {len(changed)}")

out = {"rep_of": rep_of, "canonical": canonical,
       "order": "_target_key = (-r,-s,|abs|) corner survivor (max)",
       "convention": "clean-den edges only (see canonicalize_GR.py)",
       "source": "results/gr_transforms.pkl (numerically verified store)"}
with open(os.path.join(ROOT, "results/canonical_sectors_GR_v2.pkl"), "wb") as f:
    pickle.dump(out, f)
with open(os.path.join(ROOT, "results/canonical_sectors_GR_v2.txt"), "w") as f:
    f.write(",".join(str(s) for s in canonical) + "\n")
print("saved -> results/canonical_sectors_GR_v2.pkl / .txt")
