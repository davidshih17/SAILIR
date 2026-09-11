#!/usr/bin/env python
"""Truth one-step for the g1017 13h straggler [1,-1,-1,1,1,1,0,1,1,1,-1,...]."""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from truth_engine import TruthEngine
import topo_config as _tc

T = (1, -1, -1, 1, 1, 1, 0, 1, 1, 1, -1, 0, 0, 0, 0)
triv = os.path.join(_tc.TOPO_DIR, "kira_validate/sectormappings/GR/trivialsector")
eng = TruthEngine(_tc.TOPO_DIR, triv)
r = eng.worker_replay(T, dr=1, ds=1)
print(f"{list(T)}: {r['steps']} truth actions -> {len(r['result_expr'])} lower terms")
with open(os.path.join(ROOT, "results/truth/g1017_straggler_onestep.pkl"), "wb") as f:
    pickle.dump({T: r}, f)
print("saved -> results/truth/g1017_straggler_onestep.pkl")
