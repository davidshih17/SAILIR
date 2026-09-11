#!/usr/bin/env python
"""L7 branching factor: how many L7 children does one L7 reduction emit?

>1 and the sector never drains. <1 and it does, at a rate this measures.
Counts DISTINCT children per parent (the cache dedupes repeats), and separates
children that merely slide down within L7 from ones that actually leave it.
"""
import os, pickle, random, sys
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
from sailir.ibp_env import is_master
from total_order import sector_mask

D=R+'/results/gr_reduce/g1023_greedy_certified/work/results'
def props(i): return bin(sector_mask(i)).count('1')

names=[n for n in os.listdir(D) if n.endswith('.pkl')]
random.Random(5).shuffle(names)
n_par=0; tot_child=0; tot_l7=0; sizes=[]; l7s=[]
lvl=Counter()
for n in names:
    if n_par>=4000: break
    try: d=pickle.load(open(os.path.join(D,n),'rb'))
    except Exception: continue
    if not d.get('success'): continue
    T=d.get('original_integral')
    if T is None: continue
    T=tuple(int(x) for x in T)
    if props(T)!=7: continue
    expr=d.get('final_expr') or {}
    ch=[i for i,c in expr.items() if c and not is_master(i)]
    if not ch and not expr: continue
    n_par+=1
    k7=sum(1 for i in ch if props(i)==7)
    tot_child+=len(ch); tot_l7+=k7
    sizes.append(len(ch)); l7s.append(k7)
    for i in ch: lvl[props(i)]+=1

if not n_par:
    print("  no L7 parents found in the sample"); sys.exit()
sizes.sort(); l7s.sort()
print(f"  L7 parents sampled: {n_par:,}")
print(f"  children per parent : mean {tot_child/n_par:7.2f}  median {sizes[len(sizes)//2]}  max {sizes[-1]}")
print(f"  L7 children per parent (BRANCHING FACTOR): mean {tot_l7/n_par:6.3f}  "
      f"median {l7s[len(l7s)//2]}  max {l7s[-1]}")
print(f"  parents emitting ZERO L7 children: {sum(1 for x in l7s if x==0):,} "
      f"({100.0*sum(1 for x in l7s if x==0)/n_par:.1f}%)")
print(f"\n  where all children land (by propagator count):")
for k in sorted(lvl, reverse=True):
    print(f"    L{k}: {lvl[k]:9,}  ({100.0*lvl[k]/max(tot_child,1):5.1f}%)")
b=tot_l7/n_par
print(f"\n  branching factor b = {b:.3f}  ->  "
      + (f"DRAINS; a population of 482 needs ~{482/max(1-b,1e-9):.0f} more L7 reductions"
         if b<1 else "NEVER DRAINS (b >= 1)"))
