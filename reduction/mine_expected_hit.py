#!/usr/bin/env python
"""EXPECTED HIT RATE of the expanded (mined) cache.

For each step of a finished worker, path stores (target, ibp_op, delta) and
seed = target + delta (greedy_reduce.py:647). The RAW identity
get_raw_equation_cached(op, seed) is an exact IBP relation on its own -- no
resolved_subs needed -- so solving it for its MAXIMAL element X yields a valid
one-level reduction of X at O(1) memory.

Measured here, per mined X:
  - is the rearrangement legal (every other term strictly lower, or terminal)?
  - was X ever dispatched as its own target (cache filenames + live queue)?
    i.e. work we PAID for that this would have supplied free.
"""
import os, pickle, random, sys, time
R='/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, R); sys.path.insert(0, R+'/reduction')
from sailir.topology import Topology
from sailir import ibp_env as ie
import topo_config as tc
ie.init_from_topology(Topology.from_dir(tc.TOPO_DIR)); ie.set_prime(101)
ie.set_paper_masters_only(False)
from canonical_masters import apply_canonical_masters; apply_canonical_masters()
from sailir.ibp_env import is_master, IBPEnvironment, solve_ibp_for
from total_order import tkey

D=R+'/results/gr_reduce/g1023_greedy_certified/work/results'
names=[n for n in os.listdir(D) if n.endswith('.pkl')]
dispatched=set()
for n in names:
    try: dispatched.add(tuple(int(x) for x in n[:-4].split('_')[2:]))
    except Exception: pass
print(f"  targets ever dispatched: {len(dispatched):,}", flush=True)

env=IBPEnvironment()
random.Random(0).shuffle(names)
NW=int(os.environ.get('NW','60'))
n_w=0; n_steps=0; t0=time.time()
mined=set(); illegal=0; nomax=0; t_raw=0.0
for n in names:
    if n_w>=NW: break
    try: d=pickle.load(open(os.path.join(D,n),'rb'))
    except Exception: continue
    if not d.get('success'): continue
    p=d.get('path') or []
    if len(p)<5: continue
    n_w+=1
    for (tgt,op,delta) in p:
        tgt=tuple(tgt); delta=tuple(delta)
        seed=tuple(tgt[i]+delta[i] for i in range(len(tgt)))
        try:
            ta=time.time(); raw=env.get_raw_equation_cached(op, seed); t_raw+=time.time()-ta
        except Exception:
            continue
        if not raw: continue
        n_steps+=1
        X=min(raw, key=tkey)                       # smallest tkey = highest
        others=[k for k in raw if k!=X]
        if not others: nomax+=1; continue
        if not all(tkey(k)>tkey(X) or is_master(k) for k in others):
            illegal+=1; continue
        mined.add(X)
print(f"  workers replayed : {n_w}   steps {n_steps:,}   raw-eq time {t_raw:.1f}s "
      f"({1000*t_raw/max(n_steps,1):.2f} ms/step)   wall {time.time()-t0:.0f}s", flush=True)
print(f"  illegal rearrangements: {illegal:,}   degenerate: {nomax:,}")
print(f"  DISTINCT mined reductions: {len(mined):,}  ({len(mined)/max(n_w,1):.1f} per worker)")
hit=mined & dispatched
print(f"  of those, ALSO dispatched as their own target: {len(hit):,} "
      f"({100*len(hit)/max(len(mined),1):.1f}%)")
print(f"  => per worker, {len(hit)/max(n_w,1):.1f} reductions that the campaign PAID a worker for")
