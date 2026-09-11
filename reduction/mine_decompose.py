#!/usr/bin/env python
"""Why was the holdout hit rate 0? Decompose the overlap.

Tests three things the holdout conflated:
  A. |mined index  n  ALL dispatched targets|   -- is there overlap at all?
  B. of that overlap, how much is ALREADY-SOLVED vs genuinely new?
  C. how many late targets were skipped as already-solved (the 0 may be structural)
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

names = sorted((os.path.getmtime(os.path.join(D, n)), n)
               for n in os.listdir(D) if n.endswith('.pkl'))
print(f"  {len(names):,} results", flush=True)
env = IBPEnvironment()

def load(n):
    try:
        with open(os.path.join(D, n), 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

index = set(); solved_early = set()
t0 = time.time()
for i, (_, n) in enumerate(names[:N_TRAIN]):
    d = load(n)
    if not d or not d.get('success'):
        continue
    t = d.get('original_integral')
    if t is not None:
        solved_early.add(tuple(int(x) for x in t))
    for (tg, op, delta) in (d.get('path') or []):
        tg = tuple(int(x) for x in tg)
        seed = tuple(tg[k] + int(delta[k]) for k in range(15))
        try:
            raw = env.get_raw_equation_cached(op, seed)
        except Exception:
            continue
        if not raw:
            continue
        X = min(raw, key=tkey)
        if all(tkey(k) > tkey(X) or is_master(k) for k in raw if k != X):
            index.add(tuple(int(x) for x in X))
print(f"  index {len(index):,}   early-solved {len(solved_early):,}  "
      f"({time.time()-t0:.0f}s)", flush=True)

# ---- A/B: overlap against EVERY dispatched target in the campaign ---------
all_t = set(); t1 = time.time()
for j, (_, n) in enumerate(names):
    d = load(n)
    if d and d.get('original_integral') is not None:
        all_t.add(tuple(int(x) for x in d['original_integral']))
    if (j + 1) % 100000 == 0:
        print(f"    scanned {j+1:,} targets  ({time.time()-t1:.0f}s)", flush=True)
print(f"  ALL dispatched targets (distinct): {len(all_t):,}", flush=True)

ov = index & all_t
print(f"  A. index n all-targets : {len(ov):,}  "
      f"({100.0*len(ov)/max(len(index),1):.1f}% of index, "
      f"{100.0*len(ov)/max(len(all_t),1):.1f}% of targets)", flush=True)
print(f"  B. of that, early-solved: {len(ov & solved_early):,}   "
      f"NOT early-solved: {len(ov - solved_early):,}", flush=True)

# ---- C: what happened to the late cohort ---------------------------------
skipped = scored = hit = 0
mined_only = index - solved_early
for _, n in names[-N_TEST:]:
    d = load(n)
    if not d or d.get('original_integral') is None:
        continue
    tg = tuple(int(x) for x in d['original_integral'])
    if tg in solved_early:
        skipped += 1; continue
    scored += 1
    if tg in mined_only:
        hit += 1
print(f"  C. late cohort: skipped(already solved) {skipped:,}  scored {scored:,}  "
      f"hit {hit:,}", flush=True)
print(f"     late targets that ARE in the raw index: "
      f"{sum(1 for _,n in names[-N_TEST:] for d in [load(n)] if d and d.get('original_integral') is not None and tuple(int(x) for x in d['original_integral']) in index):,}",
      flush=True)
