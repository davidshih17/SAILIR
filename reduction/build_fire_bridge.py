#!/usr/bin/env python
"""Change-of-basis bridge: reduce every FIRE-oracle master that is NOT in our
142 basis fully to OUR masters with the truth engine. Writes
results/fire_bridge_142.pkl = {fire_master: {our_master: coeff}}."""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from truth_engine import TruthEngine
import topo_config as _tc
from sailir import ibp_env

ours = set()
for ln in open(os.path.join(ROOT, "results/gr_cut_masters_142.txt")):
    ours.add(tuple(int(x) for x in ln.split(",")))

oracle = pickle.load(open(os.path.join(ROOT, "results/fire_oracle_GR.pkl"), "rb"))
fire_masters = set()
for expr in oracle.values():
    fire_masters.update(tuple(k) for k in expr)
need = sorted(fire_masters - ours)
print(f"FIRE masters appearing in oracle: {len(fire_masters)}; "
      f"not in our 142 basis: {len(need)}")

triv = os.path.join(_tc.TOPO_DIR, "kira_validate/sectormappings/GR/trivialsector")
eng = TruthEngine(_tc.TOPO_DIR, triv)
bridge = {}
for T in need:
    print(f"reducing {list(T)} ...", flush=True)
    r = eng.reduce(T, dr=1, ds=1)
    bad = [k for k in r["final_expr"] if k not in ours]
    assert not bad, f"reduction of {list(T)} left non-142 terms: {bad[:3]}"
    bridge[T] = dict(r["final_expr"])
    print(f"  -> {len(bridge[T])} terms of the 142 basis "
          f"({r['steps']} steps, {r['time']:.0f}s)", flush=True)
    eng.systems.clear()

with open(os.path.join(ROOT, "results/fire_bridge_142.pkl"), "wb") as f:
    pickle.dump(bridge, f)
print(f"saved {len(bridge)} bridge entries -> results/fire_bridge_142.pkl")
