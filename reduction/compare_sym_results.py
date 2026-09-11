#!/usr/bin/env python
"""Compare two onestep_worker result pkls (symmetry off vs on): success, step
counts (model vs symmetry), time, and the CORRECTNESS check — the master-only
part of final_expr must be identical."""
import sys, os, pickle
sys.path.insert(0, "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2")
sys.path.insert(0, "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/reduction")
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import set_prime, set_paper_masters_only, is_master

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
topo = Topology.from_dir(os.path.join(ROOT, "topology_input/pentagonbox"))
ibp_env.init_from_topology(topo); set_prime(1009); set_paper_masters_only(True)

off = pickle.load(open(sys.argv[1], "rb"))
on = pickle.load(open(sys.argv[2], "rb"))

def steps(r):
    p = r["path"]
    sym = sum(1 for e in p if e[1] == "__SYM__")
    return len(p), sym, len(p) - sym   # total, symmetry, model

def masters_only(expr):
    return {k: v for k, v in expr.items() if is_master(k)}

for name, r in (("OFF", off), ("ON ", on)):
    tot, sym, mod = steps(r)
    print(f"{name}: success={r['success']}  steps total={tot}  model={mod}  symmetry={sym}  "
          f"time={r['time']:.1f}s  peak={r.get('peak_memory_kb',0)/1024:.0f}MB")

mo_off = masters_only(off["final_expr"])
mo_on = masters_only(on["final_expr"])
match = mo_off == mo_on
print(f"\nCORRECTNESS: master-decomposition identical? {match}  "
      f"({len(mo_off)} vs {len(mo_on)} masters)")
if not match:
    only_off = {k: v for k, v in mo_off.items() if mo_on.get(k) != v}
    only_on = {k: v for k, v in mo_on.items() if mo_off.get(k) != v}
    print(f"  MISMATCH off-only/diff: {list(only_off)[:4]}")
    print(f"  MISMATCH on-only/diff : {list(only_on)[:4]}")
else:
    om, mm = steps(off)[2], steps(on)[2]
    if mm:
        print(f"\n  model steps: {om} -> {mm}  ({om/max(1,mm):.2f}x fewer model steps with symmetry)")
    print(f"  wall-clock : {off['time']:.1f}s -> {on['time']:.1f}s")
