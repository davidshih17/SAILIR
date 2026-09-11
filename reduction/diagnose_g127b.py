#!/usr/bin/env python
"""Corrected router-completeness check for T = g127's integral: the EQUAL-VALUE
orbit is the closure under SINGLE-TERM images only (I[K] = co * I[J]); debris
terms of multi-term images are not value-equal partners. T is correctly a
survivor iff it is the orbit minimum in the total order AND the monolithic RREF
gives it no strictly-lower rewrite (already reported SURVIVOR)."""
import os, sys, pickle
os.environ['SAILIR_TOPOLOGY'] = 'gravity3L'
os.environ['SAILIR_SECTOR_RANK'] = '1'
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc
ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
ibp_env.set_prime(1009)
from symmetry_route import tkey
CG = _tc.canonicalize_module()

T = (1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0)
seen = {T}
frontier = [T]
while frontier:
    K = frontier.pop()
    for (M, c) in CG._transforms(K):
        img = CG.image_unsigned(K, M, c)
        if img is None or len(img) != 1:
            continue
        (J, co), = img.items()
        if J not in seen:
            seen.add(J)
            frontier.append(J)
orb = sorted(seen, key=tkey)
lowest = max(seen, key=tkey)          # largest tkey = lowest in the ordering
print(f"equal-value orbit size: {len(seen)}")
print(f"orbit members (order high->low): "
      f"{[list(x) for x in orb[:4]]}{' ...' if len(orb) > 4 else ''}")
print(f"lowest-in-order member: {list(lowest)}")
print("T is the orbit minimum — SURVIVOR verdict CORRECT"
      if lowest == T else
      f"T is NOT the orbit minimum — router should have rewritten to {list(lowest)}")
