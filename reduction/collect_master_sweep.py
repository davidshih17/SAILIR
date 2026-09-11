#!/usr/bin/env python
"""Collect the master sweep: per sector compare depth-A vs depth-B master
lists (stability gate), union everything, cross-check against the canonical-45
and Kira-sym-64 lists, and write results/gr_complete_masters.txt."""
import os, re, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
OUT = os.path.join(ROOT, "results/gr_master_sweep")
G = os.path.join(ROOT, "topology_input/gravity3L")


def read_masters(d):
    p = os.path.join(d, "results/GR/masters")
    if not os.path.exists(p):
        return None
    out = set()
    for ln in open(p):
        m = re.match(r"GR\[([0-9,\-]+)\]", ln.strip())
        if m:
            out.add(tuple(int(x) for x in m.group(1).split(",")))
    return out

dirs = [l.strip() for l in open(os.path.join(OUT, "sweep_dirs.txt")) if l.strip()]
sectors = sorted({int(re.search(r"S(\d+)_", d).group(1)) for d in dirs})

union = set()
unstable = []
missing = []
per_sector = {}
for S in sectors:
    a = read_masters(os.path.join(OUT, f"S{S}_A"))
    b = read_masters(os.path.join(OUT, f"S{S}_B"))
    if a is None or b is None:
        missing.append(S)
        continue
    # keep only THIS sector's masters (subsector masters appear in deeper jobs)
    a = {m for m in a if sum(1 << i for i in range(10) if m[i] > 0) == S}
    b = {m for m in b if sum(1 << i for i in range(10) if m[i] > 0) == S}
    if a != b:
        unstable.append((S, sorted(a - b), sorted(b - a)))
    per_sector[S] = sorted(b)          # deeper depth wins
    union |= b

print(f"sectors collected: {len(per_sector)}/{len(sectors)}; "
      f"missing/incomplete: {len(missing)} {missing[:10]}")
print(f"UNSTABLE sectors (A vs B master lists differ): {len(unstable)}")
for S, onlyA, onlyB in unstable[:10]:
    print(f"   S{S}: only-A {onlyA[:3]} only-B {onlyB[:3]}")
print(f"union of sector masters: {len(union)}")

# cross-checks
import importlib.util, sys
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
os.environ.setdefault('SAILIR_TOPOLOGY', 'gravity3L')
os.environ.setdefault('SAILIR_SECTOR_RANK', '1')
from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc
ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
ibp_env.set_prime(1009)
ibp_env.set_paper_masters_only(True)
from canonical_masters import apply_canonical_masters
apply_canonical_masters()
ours45 = {tuple(m) for m in ibp_env.MASTERS_SET}
missing45 = ours45 - union
print(f"canonical-45 contained in union: {len(ours45 & union)}/45 "
      f"(missing: {[list(m) for m in sorted(missing45)][:5]})")

sym64 = set()
for ln in open(os.path.join(G, "kira_validate/results/GR/masters")):
    m = re.match(r"GR\[([0-9,\-]+)\]", ln.strip())
    if m:
        sym64.add(tuple(int(x) for x in m.group(1).split(",")))
# sym-64 masters live in Kira's preferred sectors; compare only those in
# canonical sectors directly, count the rest as needing orbit mapping
cm = pickle.load(open(os.path.join(ROOT, "results/canonical_sectors_GR_v2.pkl"), "rb"))
CANON = set(cm["canonical"])
sym_in_canon = {m for m in sym64
                if sum(1 << i for i in range(10) if m[i] > 0) in CANON}
print(f"Kira-sym-64: {len(sym_in_canon)} live in canonical sectors; "
      f"contained in union: {len(sym_in_canon & union)}"
      f" (missing: {[list(m) for m in sorted(sym_in_canon - union)][:5]})")

with open(os.path.join(ROOT, "results/gr_complete_masters.txt"), "w") as f:
    for m in sorted(union):
        f.write(",".join(str(x) for x in m) + "\n")
print(f"saved -> results/gr_complete_masters.txt ({len(union)} masters)")
print("STABILITY: " + ("ALL PASS" if not unstable and not missing else "SEE ABOVE"))
