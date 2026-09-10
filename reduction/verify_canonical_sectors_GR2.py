#!/usr/bin/env python
"""Behavioral gate step 1: rebuild the GR sector orbits with the ACTIVE
canonicalize provider (topo_config — under the A/B override this is
canonicalize2 + the general-engine store) and require EXACT agreement with the
production canonical_sectors_GR_v2.pkl (clean-den convention)."""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
import topo_config as _tc
CG = _tc.canonicalize_module()
print(f"provider: {_tc.CANONICALIZE_MOD}")

N_DEN, N, P = CG.N_DEN, CG.N, CG.P


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
        if img is not None and len(img) == 1:
            (J, co), = img.items()
            if co % P == 1:
                out.add(sec(J))
    edges[m] = out

rep_of = {}
seen = set()
for S in range(1, 1 << N_DEN):
    if S in seen:
        continue
    orb = {S}; fr = [S]
    while fr:
        T = fr.pop()
        for U in edges[T]:
            if U not in orb:
                orb.add(U); fr.append(U)
    rep = sec(max((corner(T) for T in orb), key=tkey))
    for T in orb:
        rep_of[T] = rep
    seen |= orb

ref = pickle.load(open(_tc.CANON_PKL, "rb"))
agree = sum(1 for S in range(1, 1 << N_DEN) if rep_of[S] == ref["rep_of"][S])
n_can = len(set(rep_of.values()))
print(f"orbits: canonical {n_can} (reference {len(ref['canonical'])}); "
      f"rep agreement {agree}/{(1 << N_DEN) - 1}")
print("ALL PASS" if agree == (1 << N_DEN) - 1 else "FAIL")
