#!/usr/bin/env python
"""ONE-STEP ground-truth action sequences for every integral the 12h+ stuck
Condor workers are grinding on. For each: the certified worker-form action
list, the strictly-lower one-step result, and a ZERO/nonzero verdict on the
result being empty. Systems persist and are shared across integrals."""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from truth_engine import TruthEngine
import topo_config as _tc

STUCK = [
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
    # hardest last (sector-635 near-corner: 5 ladder rungs exhausted in the
    # first attempt; its persisted systems reload from disk)
    "1,1,0,1,1,1,1,0,0,1,-1,0,0,0,0",
]

triv = os.path.join(_tc.TOPO_DIR, "kira_validate/sectormappings/GR/trivialsector")
eng = TruthEngine(_tc.TOPO_DIR, triv)
out = {}
for spec in STUCK:
    T = tuple(int(x) for x in spec.split(","))
    try:
        r = eng.worker_replay(T, dr=1, ds=1)
        n = len(r["result_expr"])
        print(f"{spec}: {r['steps']} truth actions -> "
              f"{'EMPTY (integral = 0 at one step)' if n == 0 else f'{n} lower terms'}",
              flush=True)
        out[T] = r
    except Exception as e:
        print(f"{spec}: FAIL {type(e).__name__}: {e}", flush=True)
        out[T] = None
    eng.systems.clear()      # evict between integrals (systems persist on disk)

ok = sum(1 for v in out.values() if v is not None)
print(f"\nSUMMARY: one-step truth for {ok}/{len(STUCK)} stuck integrals")
with open(os.path.join(ROOT, "results/truth/stuck_onestep_actions.pkl"), "wb") as f:
    pickle.dump(out, f)
print("saved -> results/truth/stuck_onestep_actions.pkl")
