#!/usr/bin/env python
import sys, os, pickle
from collections import Counter
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
ND=8
def sector(i): return tuple(k for k in range(ND) if i[k]>0)
def corner_of(sec): return tuple(1 if k in sec else 0 for k in range(11))
def sector_symmetric(sec):
    corner=corner_of(sec); csec=tuple(1 if k in sec else 0 for k in range(ND))
    autos=[t for t in _transforms(corner)
           if (lambda im: im and len(im)==1 and tuple(1 if x>0 else 0 for x in next(iter(im)))==csec)(image_unsigned(corner,*t))]
    return len(autos)>1
# sanity: known-symmetric sectors from earlier runs
print("SANITY (should be True):")
for s in [(1,2,4,7),(0,3,4,5),(0,1,2,3),(2,4,5,7)]:
    print(f"  sector_symmetric({s}) = {sector_symmetric(s)}")
# distinct passenger/touched sectors
runs=["probe_84_v7_successonly","probe_74_v7_successonly","probe_monster_v7_totalweight","probe_memhog_v7_totalweight"]
touched=set()
for r in runs:
    with open(f"results/{r}/result.pkl","rb") as f: d=pickle.load(f)
    for k in list(d.get('full_subs_replay',{}))+list(d.get('full_expr_replay',{})): touched.add(tuple(k))
secs=Counter(sector(I) for I in touched)
print(f"\n{len(secs)} distinct sectors touched (all 4 runs). By prop-count:")
bypc=Counter(len(s) for s in secs)
for p in sorted(bypc): print(f"  {p}-prop: {bypc[p]} distinct sectors")
print("\nall distinct 4- and 5-prop sectors touched (symmetric?):")
for s in sorted(secs, key=lambda x:(len(x),x)):
    if len(s) in (4,5):
        print(f"  {list(s)} ({len(s)}p) x{secs[s]}  symmetric={sector_symmetric(s)}")
