#!/usr/bin/env python
"""Per-integral symmetry-route solve timing on THIS node.

Samples integrals that the async replay actually had to fresh-solve (its
bench sym_memo minus the launch table = the solved set), times
canonical_monolithic_rule on each, prints the distribution.
"""
import os
import sys
import time
import pickle

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))

from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc
ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
ibp_env.set_prime(1009)
ibp_env.set_paper_masters_only(True)
from canonical_masters import apply_canonical_masters
apply_canonical_masters()
from symmetry_route import canonical_monolithic_rule

with open(ROOT + "/results/orch_bench/work_fid_async/sym_memo.pkl", "rb") as f:
    solved_tab = pickle.load(f)
with open(ROOT + "/results/gr_reduce/g1023/work/sym_memo_raw_backup.pkl",
          "rb") as f:
    launch_tab = pickle.load(f)
fresh = [k for k in solved_tab if k not in launch_tab]
print(f"fresh-solved pool: {len(fresh)} integrals", flush=True)

N = 60
step = max(1, len(fresh) // N)
sample = fresh[::step][:N]
times = []
for i, I in enumerate(sample):
    t0 = time.time()
    canonical_monolithic_rule(I)
    dt = time.time() - t0
    times.append(dt)
    print(f"{i+1:3d}/{len(sample)}  {dt:8.2f}s  {list(I)}", flush=True)

times.sort()
n = len(times)
print(f"\nn={n}  mean={sum(times)/n:.2f}s  med={times[n//2]:.2f}s  "
      f"p90={times[int(n*.9)]:.2f}s  max={times[-1]:.2f}s  "
      f"min={times[0]:.3f}s", flush=True)
