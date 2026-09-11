#!/usr/bin/env python
"""Does the mined index cover integrals the WORKERS could not solve?

The memory watcher killed 300 distinct L7 integrals (and 1,187 in total) at
15 GB. A mined recipe for one of those is a reduction obtained as a byproduct
of some OTHER walk -- work already paid for -- that no worker was ever able to
finish directly. Each hit is a permanently-blocked frontier integral resolved
for 0.2 ms of verification.
"""
import json, os, sys
R='/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0,R); sys.path.insert(0,R+'/reduction')
from sailir.topology import Topology
from sailir import ibp_env as ie
import topo_config as tc
ie.init_from_topology(Topology.from_dir(tc.TOPO_DIR))
ie.set_prime(int(os.environ.get('SAILIR_PRIME','101')))
ie.set_paper_masters_only(False)
from canonical_masters import apply_canonical_masters; apply_canonical_masters()
from sailir.ibp_env import is_master, IBPEnvironment, solve_ibp_for
from total_order import tkey, sector_mask
from recipe_index import RecipeIndex

idx = RecipeIndex(sys.argv[1])
print(f"  index loaded and SORT-VERIFIED: {len(idx):,} recipes")
env = IBPEnvironment()

P=R+'/results/gr_reduce/g1023_greedy_certified/work/diverged_killed.jsonl'
dead=set()
for line in open(P):
    line=line.strip()
    if not line: continue
    try: d=json.loads(line)
    except Exception: continue
    I=d.get('integral')
    if not isinstance(I,str): continue
    dead.add(tuple(int(x) for x in I.split('_')))
print(f"  distinct integrals killed by the memory watcher: {len(dead):,}")

def props(i): return bin(sector_mask(i)).count('1')
by={}
hit=ver=0
for I in dead:
    L=props(I)
    b=by.setdefault(L,[0,0,0])
    b[0]+=1
    if idx.get(I) is not None:
        b[1]+=1; hit+=1
        rule=idx.resolve(I, env, is_master, tkey, solve_ibp_for)
        if rule is not None:
            b[2]+=1; ver+=1
print(f"\n  {'level':>6} {'dead':>7} {'in index':>9} {'VERIFIED':>9}")
for L in sorted(by, reverse=True):
    n,h,v=by[L]
    print(f"  {'L'+str(L):>6} {n:7,} {h:9,} {v:9,}   ({100.0*v/max(n,1):.1f}%)")
print(f"\n  TOTAL dead {len(dead):,}  ->  in index {hit:,}  ->  "
      f"VERIFIED LEGAL {ver:,}  ({100.0*ver/max(len(dead),1):.1f}%)")
print("  "+idx.stats())
