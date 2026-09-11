#!/usr/bin/env python
"""Dump the truth closure LIBRARY for one target as a path-independent set.

WHY THIS IS THE RIGHT OBJECT. The truth engine's one-step closure is a SET of
(op, seed) actions -- the dependency closure of the target rule. Which order
they get consumed in is a free choice (first/last/random all reach success in
338/352/352 steps), so membership in the set does NOT depend on the path taken.

That is what makes the beam measurement well-posed. Earlier tracking followed a
single LEX path and asked whether the beam still contained it, which fails for
two reasons: lex is one arbitrary tie-break among ~18 equally-correct rows per
step, and matching a path prefix conflates states (a state is (expr,
resolved_subs), not an expression). Membership in this set has neither problem
-- a beam state is "still inside the library" iff EVERY action on its path was
drawn from here, which is a local test on the path with no matching at all.

Output: JSON {"integral": [...], "library": [[op, seed...], ...], "n": N}.
"""
import os
import sys
import json
import argparse

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))
sys.path.insert(0, os.path.join(ROOT, "data-gen"))

from sailir.topology import Topology
import topo_config as _tc
import generate_multisector_data as gm
from truth_engine import TruthEngine, sector_of

ap = argparse.ArgumentParser()
ap.add_argument('--integral', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--dr', type=int, default=1)
ap.add_argument('--ds', type=int, default=1)
a = ap.parse_args()

T = tuple(int(x) for x in a.integral.strip("'\"").split(","))
topology = Topology.from_dir(_tc.TOPO_DIR)
gm.init_from_topology(topology)
gm.PRIME = 1009

eng = TruthEngine(_tc.TOPO_DIR, os.path.join(
    _tc.TOPO_DIR, "kira_validate/sectormappings/GR/trivialsector"))
wr = eng.worker_replay(T, dr=a.dr, ds=a.ds, verbose=True)

lib = [[int(op)] + [int(x + d) for x, d in zip(J, delta)]
       for (J, op, delta) in wr['actions']]
with open(a.output, 'w') as f:
    json.dump(dict(integral=list(T), library=lib, n=len(lib)), f)
print(f"LIBRARY target={a.integral} n_actions={len(lib)} "
      f"sector={sector_of(T)} -> {a.output}", flush=True)
