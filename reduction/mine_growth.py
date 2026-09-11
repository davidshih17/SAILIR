#!/usr/bin/env python
"""Does the mined set SATURATE, or does it explode linearly with workers?

Mines an increasing number of workers and reports the DISTINCT mined integrals.
Linear growth => cache explodes ~94x. Sublinear => the set is bounded and the
blow-up factor is far smaller.
"""
import os, pickle, random, sys, time
R='/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0,R); sys.path.insert(0,R+'/reduction')
from sailir.topology import Topology
from sailir import ibp_env as ie
import topo_config as tc
ie.init_from_topology(Topology.from_dir(tc.TOPO_DIR)); ie.set_prime(101)
ie.set_paper_masters_only(False)
from canonical_masters import apply_canonical_masters; apply_canonical_masters()
from sailir.ibp_env import is_master, IBPEnvironment
from total_order import tkey

D=R+'/results/gr_reduce/g1023_greedy_certified/work/results'
names=[n for n in os.listdir(D) if n.endswith('.pkl')]
dispatched=set()
for n in names:
    try: dispatched.add(tuple(int(x) for x in n[:-4].split('_')[2:]))
    except Exception: pass
random.Random(0).shuffle(names)
env=IBPEnvironment()
mined=set(); n_w=0; steps=0; t0=time.time()
marks=[25,50,100,200,400,800]
print(f"  dispatched targets: {len(dispatched):,}", flush=True)
print(f"  {'workers':>8} {'steps':>9} {'distinct mined':>15} {'new/worker':>11} {'hit dispatched':>15} {'wall':>7}", flush=True)
for n in names:
    if n_w>=max(marks): break
    try: d=pickle.load(open(os.path.join(D,n),'rb'))
    except Exception: continue
    if not d.get('success'): continue
    p=d.get('path') or []
    if len(p)<5: continue
    n_w+=1
    for (tgt,op,delta) in p:
        tgt=tuple(tgt); delta=tuple(delta)
        seed=tuple(tgt[i]+delta[i] for i in range(len(tgt)))
        try: raw=env.get_raw_equation_cached(op,seed)
        except Exception: continue
        if not raw: continue
        steps+=1
        X=min(raw,key=tkey)
        others=[k for k in raw if k!=X]
        if others and all(tkey(k)>tkey(X) or is_master(k) for k in others):
            mined.add(X)
    if n_w in marks:
        hit=len(mined & dispatched)
        print(f"  {n_w:8d} {steps:9,} {len(mined):15,} {len(mined)/n_w:11.1f} "
              f"{hit:15,} {time.time()-t0:6.0f}s", flush=True)
