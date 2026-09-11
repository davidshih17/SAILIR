#!/usr/bin/env python
"""TEMPORAL HOLDOUT: would the recipe index have spared later worker jobs?

Build the index from the workers that finished FIRST, then ask what fraction of
the targets dispatched LATER are already covered. A covered target is a Condor
job that never needed to run. The split removes self-reference -- a worker's own
target is trivially in its own walk.
"""
import os, pickle, sys, time
R = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, R); sys.path.insert(0, R + '/reduction')
from sailir.topology import Topology
from sailir import ibp_env as ie
import topo_config as tc
ie.init_from_topology(Topology.from_dir(tc.TOPO_DIR))
ie.set_prime(int(os.environ.get('SAILIR_PRIME', '101')))
ie.set_paper_masters_only(False)
from canonical_masters import apply_canonical_masters; apply_canonical_masters()
from sailir.ibp_env import is_master, IBPEnvironment
from total_order import tkey

D = R + '/results/gr_reduce/g1023_greedy_certified/work/results'
N_TRAIN = int(os.environ.get('NTRAIN', '20000'))
N_TEST = int(os.environ.get('NTEST', '5000'))

print("  sorting results by finish time ...", flush=True)
names = sorted((os.path.getmtime(os.path.join(D, n)), n)
               for n in os.listdir(D) if n.endswith('.pkl'))
print(f"  {len(names):,} results  "
      f"[{time.strftime('%m-%d %H:%M', time.localtime(names[0][0]))} .. "
      f"{time.strftime('%m-%d %H:%M', time.localtime(names[-1][0]))}]", flush=True)

train = names[:N_TRAIN]
test = names[-N_TEST:]
env = IBPEnvironment()

def load(n):
    try:
        with open(os.path.join(D, n), 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

# ---- build the index from the EARLY workers ------------------------------
index = set()
solved_early = set()          # targets the early workers themselves solved
t0 = time.time()
for i, (_, n) in enumerate(train):
    d = load(n)
    if not d or not d.get('success'):
        continue
    tgt = d.get('original_integral')
    if tgt is not None:
        solved_early.add(tuple(tgt))
    for (t, op, delta) in (d.get('path') or []):
        t = tuple(t); seed = tuple(t[k] + delta[k] for k in range(15))
        try:
            raw = env.get_raw_equation_cached(op, seed)
        except Exception:
            continue
        if not raw:
            continue
        X = min(raw, key=tkey)
        if all(tkey(k) > tkey(X) or is_master(k) for k in raw if k != X):
            index.add(X)
    if (i + 1) % 5000 == 0:
        print(f"    train {i+1:6,}  index {len(index):9,}  "
              f"{time.time()-t0:4.0f}s", flush=True)

mined_only = index - solved_early
print(f"  EARLY workers solved      : {len(solved_early):,} targets", flush=True)
print(f"  index from their walks    : {len(index):,} integrals", flush=True)
print(f"  ... NEW beyond the targets: {len(mined_only):,} "
      f"({len(mined_only)/max(len(solved_early),1):.1f}x)", flush=True)

# ---- score the LATE dispatches -------------------------------------------
hit = miss = 0; hit_steps = miss_steps = 0
for _, n in test:
    d = load(n)
    if not d or d.get('original_integral') is None:
        continue
    tgt = tuple(d['original_integral'])
    if tgt in solved_early:
        continue                      # already cached the normal way, not our win
    st = d.get('steps') or 0
    if tgt in mined_only:
        hit += 1; hit_steps += st
    else:
        miss += 1; miss_steps += st
tot = hit + miss
print(f"  LATE dispatches scored    : {tot:,}", flush=True)
if tot:
    print(f"  covered by the MINED index: {hit:,}  ({100.0*hit/tot:.1f}%)", flush=True)
    print(f"  search steps those cost   : {hit_steps:,} "
          f"(avg {hit_steps/max(hit,1):.1f}/target)", flush=True)
    print(f"  avg steps of the MISSES   : {miss_steps/max(miss,1):.1f}", flush=True)
