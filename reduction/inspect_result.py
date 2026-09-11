#!/usr/bin/env python
"""Inspect the terms a worker produced: where do the non-masters SIT?

The question is whether a one-level reduction is actually descending much, or
just shuffling weight sideways -- i.e. whether the 62 non-masters are a real
step down or nearly-as-hard integrals that each need their own worker.
"""
import os, pickle, sys
from collections import Counter
R='/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0,R); sys.path.insert(0,R+'/reduction')
from sailir.topology import Topology
from sailir import ibp_env as ie
import topo_config as tc
ie.init_from_topology(Topology.from_dir(tc.TOPO_DIR))
ie.set_prime(int(os.environ.get('SAILIR_PRIME','101')))
ie.set_paper_masters_only(False)
from canonical_masters import apply_canonical_masters; apply_canonical_masters()
from sailir.ibp_env import is_master, weight
from total_order import tkey, sector_mask, base_key

f = sys.argv[1]
d = pickle.load(open(f,'rb'))
T = tuple(int(x) for x in d['original_integral'])
expr = d.get('final_expr') or {}
print(f"  target      I{list(T)}")
print(f"  success={d.get('success')}  steps={d.get('steps')}  "
      f"time={d.get('time',0):.0f}s  peak_mem={d.get('peak_memory_kb',0)/1e6:.2f} GB")
print(f"  start_w12={d.get('start_w12')}  best_max_w12={d.get('best_max_w12')}  "
      f"best_n_non_masters={d.get('best_n_non_masters')}")
print(f"  |final_expr| = {len(expr):,} terms")

def prof(i):
    r = sum(x for x in i if x > 0)
    s = -sum(x for x in i if x < 0)
    nprop = bin(sector_mask(i)).count('1')
    return nprop, r, s

nm = [i for i,c in expr.items() if c and not is_master(i)]
ms = [i for i,c in expr.items() if c and is_master(i)]
print(f"  masters {len(ms):,}   non-masters {len(nm):,}")
pT = prof(T); kT = tkey(T)
print(f"\n  target profile: props={pT[0]} r={pT[1]} s={pT[2]}  weight={weight(T)}")

above = sum(1 for i in nm if tkey(i) < kT)
below = sum(1 for i in nm if tkey(i) > kT)
same  = sum(1 for i in nm if tkey(i) == kT)
print(f"  non-masters ABOVE target (tkey <): {above}")
print(f"  non-masters BELOW target (tkey >): {below}")
print(f"  non-masters EQUAL                : {same}")

c = Counter(prof(i) for i in nm)
print(f"\n  non-master (props, r, s) distribution -- {len(c)} distinct profiles:")
for k in sorted(c, reverse=True):
    d_props = k[0]-pT[0]; d_r = k[1]-pT[1]; d_s = k[2]-pT[2]
    print(f"    props={k[0]:2d} r={k[1]:2d} s={k[2]:2d}   x{c[k]:3d}   "
          f"(vs target: props{d_props:+d} r{d_r:+d} s{d_s:+d})")

print(f"\n  the 10 non-masters closest to the target (hardest remaining):")
for i in sorted(nm, key=tkey)[:10]:
    p = prof(i)
    print(f"    I{list(i)}  props={p[0]} r={p[1]} s={p[2]}")
