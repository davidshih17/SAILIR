#!/usr/bin/env python
"""For every integral in a REAL baseline reduction (its IBP one-step is in cache),
compare symmetry's free reduction to IBP's searched reduction. Classify:
  - survivor            : symmetry can't reduce -> IBP needed anyway
  - WIN (compact+safe)  : symmetry rule has <= IBP terms AND is all coarse-lower (no
                          lateral -> cycle-safe, no proliferation) -> free, no search
  - worse/lateral       : symmetry reduces but bigger and/or has lateral terms"""
import pickle, sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from symmetry_route import symmetry_rule
def w12(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0))
# a real baseline reduction cache = real integrals that occur in a reduction
d=pickle.load(open(BASE+"/results/ab_symmetry/m2_4prop_dots/baseline/reduction.pkl","rb"))
cache=d['cache']
ints=list(cache.keys())
n=len(ints)
survivor=win=worse_big=worse_lateral=0
sym_terms=[]; ibp_terms=[]
for I in ints:
    t_ibp=len([1 for v in cache[I].values() if v])
    r=symmetry_rule(I)
    if r is None:
        survivor+=1; continue
    t_sym=len(r)
    wI=w12(I)
    all_coarse_lower=all(w12(k)<wI for k in r) if r else True
    sym_terms.append(t_sym); ibp_terms.append(t_ibp)
    if t_sym<=t_ibp and all_coarse_lower:
        win+=1
    elif not all_coarse_lower:
        worse_lateral+=1
    else:
        worse_big+=1
print(f"real baseline reduction: {n} integrals (each with its IBP one-step)\n")
print(f"  survivor  (no symmetry rule; IBP needed)                : {survivor:4d}  ({100*survivor/n:.0f}%)")
print(f"  WIN       (symmetry <= IBP terms AND all coarse-lower)  : {win:4d}  ({100*win/n:.0f}%)  <- free, cycle-safe")
print(f"  worse-big (symmetry reduces but MORE terms than IBP)    : {worse_big:4d}  ({100*worse_big/n:.0f}%)")
print(f"  lateral   (symmetry rule has lateral terms -> cycle risk): {worse_lateral:4d}  ({100*worse_lateral/n:.0f}%)")
if sym_terms:
    import statistics as st
    print(f"\n  where symmetry fires: median terms sym={st.median(sym_terms):.0f} vs ibp={st.median(ibp_terms):.0f}")
