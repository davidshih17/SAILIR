#!/usr/bin/env python
"""What do routing rules actually contain? For a sample of persisted rules:
- sector of the source integral vs sectors on the right-hand side
- does the RHS stay in one target sector, or spill into lower sectors?
- rule size distribution
Sector = bitmask of propagator slots with exponent > 0 (gravity3L layout
taken from ibp_env's topology).
"""
import pickle
import sys

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/reduction")

from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc
ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
from sailir.ibp_env import sector_bitmask as sector


with open(ROOT + "/results/gr_reduce/g1023/work/sym_memo.pkl", "rb") as f:
    tab = pickle.load(f)
rules = [(k, v) for k, v in tab.items() if v]
print(f"{len(tab)} entries, {len(rules)} non-trivial rules", flush=True)

step = max(1, len(rules) // 2000)
same_sector = cross_only = with_lower = 0
sizes = []
n_lower_terms = 0
n_terms = 0
for k, v in rules[::step][:2000]:
    ks = sector(k)
    vsecs = {sector(t) for t in v}
    sizes.append(len(v))
    n_terms += len(v)
    n_lower_terms += sum(1 for t in v
                         if bin(sector(t)).count('1') < bin(ks).count('1'))
    if vsecs == {ks}:
        same_sector += 1
    elif any(bin(s2).count('1') < bin(ks).count('1') for s2 in vsecs):
        with_lower += 1
    else:
        cross_only += 1

sizes.sort()
n = len(sizes)
print(f"sample {n}: RHS entirely within SOURCE sector: {same_sector}", flush=True)
print(f"  RHS in other sector(s), same propagator count only: {cross_only}", flush=True)
print(f"  RHS containing LOWER-sector (fewer-propagator) terms: {with_lower}", flush=True)
print(f"  lower-sector terms overall: {n_lower_terms}/{n_terms}", flush=True)
print(f"rule size: med={sizes[n//2]} p90={sizes[int(n*.9)]} max={sizes[-1]}", flush=True)
