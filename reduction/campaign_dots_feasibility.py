#!/usr/bin/env python
"""For each hard dots-only campaign integral: the truth engine's projected
seed count at each rung of its escalation ladder (pure enumeration count, no
build) -> classify truth-feasible (some rung <= 25k budget could hold a rule)
vs specialist-needed. Mirrors TruthEngine.worker_replay's ladder exactly."""
import sys

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/reduction")
import topo_config as _tc
from truth_engine import TruthEngine, sector_of, rs_of

eng = TruthEngine(_tc.TOPO_DIR)
import subprocess
out = subprocess.run(
    ["/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python",
     ROOT + "/reduction/campaign_dots_levels.py"],
    capture_output=True, text=True).stdout
targets = []
for ln in out.splitlines():
    ln = ln.strip()
    if ln.startswith("L=") and "  " in ln:
        parts = ln.split()
        targets.append(tuple(int(x) for x in parts[2].split(",")))

LADDER = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (2, 2), (3, 2), (4, 2)]
print(f"{len(targets)} hard dots-only integrals; budget 25000 seeds/system")
n_feas = 0
for T in targets:
    S = sector_of(T)
    r, s = rs_of(T)
    L = bin(S).count("1")
    counts = []
    for (a, b) in LADDER:
        counts.append(sum(1 for _ in eng._seeds(S, r + a, s + b)))
    feas = [c for c in counts if c <= 25000]
    verdict = "TRUTH-FEASIBLE" if feas else "SPECIALIST"
    if feas:
        n_feas += 1
    print(f"  L={L} d={r - L} {','.join(map(str, T))}: "
          f"ladder seeds {counts[0]}..{max(counts)} "
          f"(max feasible rung {max(feas) if feas else 0}) -> {verdict}")
print(f"\n{n_feas}/{len(targets)} hard dots-only integrals are within the "
      f"truth engine's budget at some ladder rung")
