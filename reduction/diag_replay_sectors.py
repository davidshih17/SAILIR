#!/usr/bin/env python
"""Is 0/285 real? For the dispatched integrals: (1) what sectors are they in, (2) are
those sectors symmetric (check the CORNER via _transforms), (3) does _transforms(I) with
ISP numerators over-filter vs _transforms(corner)+image_unsigned(I)?"""
import sys, os, pickle, itertools
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

runs=["probe_84_v7_successonly","probe_74_v7_successonly","probe_monster_v7_totalweight","probe_memhog_v7_totalweight"]
pop=set()
for r in runs:
    with open(f"results/{r}/result.pkl","rb") as f: d=pickle.load(f)
    for k in d.get('full_subs_replay',{}): pop.add(tuple(k))

# distinct sectors among dispatched
secs=Counter(sector(I) for I in pop)
print(f"{len(pop)} dispatched integrals span {len(secs)} distinct sectors\n")

# is each sector symmetric?  (corner has nontrivial invertible autos AND image_unsigned nontrivial)
def sector_symmetric(sec):
    corner=corner_of(sec); csec=tuple(1 if k in sec else 0 for k in range(ND))
    autos=[t for t in _transforms(corner)
           if (lambda im: im and len(im)==1 and tuple(1 if x>0 else 0 for x in next(iter(im)))==csec)(image_unsigned(corner,*t))]
    return len(autos)>1
symsec={s:sector_symmetric(s) for s in secs}
n_in_sym=sum(cnt for s,cnt in secs.items() if symsec[s])
print(f"dispatched integrals in SYMMETRIC sectors: {n_in_sym} / {len(pop)}")
print(f"distinct symmetric sectors touched: {sum(symsec.values())} / {len(secs)}\n")
print("sector breakdown (prop-count, symmetric?):")
for s,cnt in secs.most_common(12):
    print(f"  {list(s)} ({len(s)}p)  x{cnt}   symmetric={symsec[s]}")

# for a dispatched integral in a symmetric sector (if any): does _transforms(I) over-filter?
print("\n=== over-filter check: _transforms(I) vs _transforms(corner)+image on I ===")
tested=0
for I in pop:
    s=sector(I)
    if not symsec[s]: continue
    corner=corner_of(s); csec=tuple(1 if k in s else 0 for k in range(ND))
    autos_c=[t for t in _transforms(corner)
             if (lambda im: im and len(im)==1 and tuple(1 if x>0 else 0 for x in next(iter(im)))==csec)(image_unsigned(corner,*t))]
    nontriv_on_I=sum(1 for t in autos_c if (lambda im: im is not None and im!={I:1})(image_unsigned(I,*t)))
    n_direct=len([t for t in _transforms(I)])
    print(f"  I={I} sector {list(s)}: corner-autos acting nontrivially on I = {nontriv_on_I}, _transforms(I) returns {n_direct}")
    tested+=1
    if tested>=5: break
if tested==0: print("  (no dispatched integral is in a symmetric sector — 0/285 is REAL)")
