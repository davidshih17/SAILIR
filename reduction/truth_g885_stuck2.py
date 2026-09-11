#!/usr/bin/env python
"""Truth one-step for the two 16h-stuck g885 workers (sector 821):
  [2,0,1,0,1,1,0,0,1,1,0,0,0,0,0]   dotted corner
  [1,0,1,0,1,1,-2,0,1,1,0,0,0,0,0]  double-numerator (proven reducible)
Writes results/truth/g885_stuck2_onestep.pkl with the certified worker-form
action sequences and the strictly-lower one-step results."""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from truth_engine import TruthEngine
import topo_config as _tc

STUCK = [
    (2, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0),
    (1, 0, 1, 0, 1, 1, -2, 0, 1, 1, 0, 0, 0, 0, 0),
]
triv = os.path.join(_tc.TOPO_DIR, "kira_validate/sectormappings/GR/trivialsector")
eng = TruthEngine(_tc.TOPO_DIR, triv)
out = {}
for T in STUCK:
    try:
        r = eng.worker_replay(T, dr=1, ds=1)
        n = len(r["result_expr"])
        print(f"{list(T)}: {r['steps']} truth actions -> "
              f"{'EMPTY' if n == 0 else f'{n} lower terms'}", flush=True)
        out[T] = r
    except Exception as e:
        print(f"{list(T)}: FAIL {type(e).__name__}: {e}", flush=True)
        out[T] = None
    eng.systems.clear()

with open(os.path.join(ROOT, "results/truth/g885_stuck2_onestep.pkl"), "wb") as f:
    pickle.dump(out, f)
print("saved -> results/truth/g885_stuck2_onestep.pkl")
