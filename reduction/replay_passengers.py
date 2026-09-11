#!/usr/bin/env python
"""Where does symmetry help start? Classify PASSENGERS (full_expr_replay = lower terms
top workers hand off) by prop-count and whether their sector is symmetric."""
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
_symcache={}
def sector_symmetric(sec):
    if sec in _symcache: return _symcache[sec]
    corner=corner_of(sec); csec=tuple(1 if k in sec else 0 for k in range(ND))
    autos=[t for t in _transforms(corner)
           if (lambda im: im and len(im)==1 and tuple(1 if x>0 else 0 for x in next(iter(im)))==csec)(image_unsigned(corner,*t))]
    r=len(autos)>1; _symcache[sec]=r; return r
runs=["probe_84_v7_successonly","probe_74_v7_successonly","probe_monster_v7_totalweight","probe_memhog_v7_totalweight"]
subs=set(); expr=set()
for r in runs:
    with open(f"results/{r}/result.pkl","rb") as f: d=pickle.load(f)
    for k in d.get('full_subs_replay',{}): subs.add(tuple(k))
    for k in d.get('full_expr_replay',{}): expr.add(tuple(k))
passengers = expr - subs
print(f"reduced (in-sector, top workers): {len(subs)}   passengers (handed off): {len(passengers)}\n")
def report(name, pop):
    bypc=Counter(); sym=Counter()
    for I in pop:
        s=sector(I); bypc[len(s)]+=1
        if sector_symmetric(s): sym[len(s)]+=1
    print(f"=== {name}: {len(pop)} integrals ===")
    tot_sym=0
    for p in sorted(bypc):
        print(f"  {p}-prop: {bypc[p]:4d}   in symmetric sector: {sym[p]:4d}")
        tot_sym+=sym[p]
    print(f"  TOTAL in symmetric sectors: {tot_sym} / {len(pop)} ({100*tot_sym/max(len(pop),1):.0f}%)\n")
report("reduced (top-worker targets)", subs)
report("passengers (next-level targets)", passengers)
report("ALL integrals touched", subs|expr)
