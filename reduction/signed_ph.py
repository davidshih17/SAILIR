#!/usr/bin/env python
"""Decisive test: does the SIGNED stabilizer projector P_H keep the corner (a genuine
master) while vanishing odd numerators? If it kills the corner, the sign is the
unphysical det-like sign the symmetries.py note warns about."""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned, RED, P   # RED = SymmetryReducer (SIGNED _image)

def secbits(i): return tuple(1 if i[k]>0 else 0 for k in range(8))
def stab(corner):
    cs=secbits(corner)
    return [(M,c) for (M,c) in _transforms(corner)
            if (lambda im: im and len(im)==1 and next(iter(im))==corner)(image_unsigned(corner,M,c))]
def PH_signed(I,H):
    acc={}
    for (M,c) in H:
        img=RED._image(I,M,c)          # SIGNED
        if img is None: continue
        for k,v in img.items(): acc[k]=(acc.get(k,0)+v)%P
    return {k:v for k,v in acc.items() if v%P}
def PH_unsigned(I,H):
    acc={}
    for (M,c) in H:
        img=image_unsigned(I,M,c)
        if img is None: continue
        for k,v in img.items(): acc[k]=(acc.get(k,0)+v)%P
    return {k:v for k,v in acc.items() if v%P}

for combo in [(0,2,4,5),(0,3,4,5),(1,2,4,7)]:
    corner=tuple(1 if k in combo else 0 for k in range(11))
    H=stab(corner)
    print("="*68); print(f"SECTOR {sorted(combo)}  |H|={len(H)}")
    tests={"corner":corner,
           "+D9^1":  tuple(corner[k]+(-1 if k==8 else 0) for k in range(11)),
           "+D9^2":  tuple(corner[k]+(-2 if k==8 else 0) for k in range(11)),
           "+D10^1": tuple(corner[k]+(-1 if k==9 else 0) for k in range(11)),
           "+D9 D10":tuple(corner[k]+(-1 if k in (8,9) else 0) for k in range(11))}
    for name,I in tests.items():
        u=PH_unsigned(I,H); s=PH_signed(I,H)
        uz=" ZERO" if not u else f"{len(u)}t"
        sz=" ZERO" if not s else f"{len(s)}t"
        flag=""
        if name=="corner" and not s: flag="  <-- corner KILLED by sign (UNPHYSICAL!)"
        if name!="corner" and u and not s: flag="  <-- vanishes only WITH sign (candidate odd-zero)"
        print(f"  {name:9s}: unsigned={uz:>6}   signed={sz:>6}{flag}")
    print()
