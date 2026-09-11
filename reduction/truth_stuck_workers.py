#!/usr/bin/env python
"""Truth-reduce every integral the 12h+ stuck Condor workers are grinding on
(all in small sectors, L=4-7): decide ZERO vs a genuine master combination.
Output: per-integral verdict + a pkl with reductions and trajectories."""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from truth_engine import TruthEngine
import topo_config as _tc

STUCK = [
    "1,1,0,1,1,1,1,0,0,1,-1,0,0,0,0",
    "1,1,0,1,1,0,1,0,0,1,0,0,0,0,0",
    "1,1,1,1,1,1,0,0,0,0,0,0,0,0,0",
    "1,2,0,1,1,0,1,0,0,1,0,0,0,0,0",
    "1,0,1,0,1,1,0,0,1,1,0,0,0,0,0",
    "1,1,1,0,1,0,1,0,1,0,0,0,0,0,0",
    "1,0,1,0,0,1,1,0,1,1,0,0,0,0,0",
    "1,1,1,1,1,1,0,-1,0,0,0,0,0,0,0",
    "2,1,1,1,1,1,0,-1,0,-1,0,0,0,0,0",
    "1,1,1,1,0,0,0,0,0,0,0,0,0,0,0",
    "1,-1,1,0,1,1,0,0,1,1,0,0,0,0,0",
    "1,1,1,1,1,1,0,0,0,-1,0,0,0,0,0",
    "2,0,1,0,1,1,0,0,1,1,0,0,0,0,0",
    "1,0,1,0,1,1,-2,0,1,1,0,0,0,0,0",
    "2,1,1,1,1,1,0,0,0,0,0,0,0,0,0",
    "1,1,1,0,-1,0,1,0,1,0,0,0,0,0,0",
    "1,1,1,0,0,-1,1,0,1,0,0,0,0,0,0",
    "1,1,-1,1,1,0,1,0,0,1,0,0,0,0,0",
    "1,1,0,1,1,0,1,-1,0,1,0,0,0,0,0",
    "1,1,0,1,1,0,1,0,-1,1,0,0,0,0,0",
]

triv = os.path.join(_tc.TOPO_DIR, "kira_validate/sectormappings/GR/trivialsector")
eng = TruthEngine(_tc.TOPO_DIR, triv)
out = {}
for s in STUCK:
    T = tuple(int(x) for x in s.split(","))
    try:
        r = eng.reduce(T, dr=1, ds=1, record_states=True)
        n = len(r["final_expr"])
        verdict = "ZERO" if n == 0 else f"{n} masters"
        print(f"{s}: {verdict} ({r['steps']} truth steps)", flush=True)
        out[T] = r
    except Exception as e:
        print(f"{s}: FAIL {type(e).__name__}: {e}", flush=True)
        out[T] = None

with open(os.path.join(ROOT, "results/truth/stuck_workers_truth.pkl"), "wb") as f:
    pickle.dump(out, f)
n_zero = sum(1 for v in out.values() if v is not None and not v["final_expr"])
n_ok = sum(1 for v in out.values() if v is not None)
print(f"\nSUMMARY: {n_ok}/{len(STUCK)} reduced; ZERO integrals: {n_zero}")
print("saved -> results/truth/stuck_workers_truth.pkl")
