#!/usr/bin/env python
"""Show, on a real symmetric sector, how the stabilizer projector P_H acts on
concrete integrals: corner, and several numerator monomials. P_H.I = (1/|H|) sum_h
image_h(I). We use the UNSIGNED action (what we have) and flag where the sign matters."""
import sys, os, fractions
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned, P
from sailir.symmetries import sector_of

def secbits(i): return tuple(1 if i[k]>0 else 0 for k in range(8))
def stab(corner):
    """corner-fixing automorphisms of the sector (image = corner)."""
    cs=secbits(corner)
    H=[(M,c) for (M,c) in _transforms(corner)
       if (lambda im: im and len(im)==1 and next(iter(im))==corner)(image_unsigned(corner,M,c))]
    return H
def PH(I,H):
    """(unnormalized) sum_h image_h(I) mod P; then we inspect the coefficient pattern."""
    acc={}
    for (M,c) in H:
        img=image_unsigned(I,M,c)
        if img is None: continue
        for k,v in img.items(): acc[k]=(acc.get(k,0)+v)%P
    return {k:v for k,v in acc.items() if v%P}

# find a symmetric sector whose automorphisms act NONTRIVIALLY on the ISP slots (8,9,10)
def isp_action(H):
    """how the automorphisms map ISP slots D9,D10,D11 (positions 8,9,10)."""
    out=[]
    for (M,c) in H:
        out.append({s: M.get(s,{}) for s in (8,9,10)})
    return out

for combo in [(0,2,4,5),(0,1,2,4),(1,2,4,7),(0,3,4,5)]:
    corner=tuple(1 if k in combo else 0 for k in range(11))
    H=stab(corner)
    if len(H)<=1: continue
    print("="*72)
    print(f"SECTOR {sorted(combo)}  (corner {list(corner)})   |stabilizer H| = {len(H)}")
    # how do the automorphisms act on the ISP slots?
    print("  automorphism action on ISP slots D9/D10/D11 (pos 8/9/10):")
    for k,act in enumerate(isp_action(H)):
        s=", ".join(f"D{sl+1}->{dict(act[sl])}" for sl in (8,9,10))
        print(f"    h{k}: {s}")
    # P_H on several integrals
    tests={
      "corner (no numerator)": corner,
      "+D9^1":  tuple(corner[k]+(-1 if k==8 else 0) for k in range(11)),
      "+D10^1": tuple(corner[k]+(-1 if k==9 else 0) for k in range(11)),
      "+D9^1 D10^1": tuple(corner[k]+(-1 if k in (8,9) else 0) for k in range(11)),
      "+D9^2": tuple(corner[k]+(-2 if k==8 else 0) for k in range(11)),
    }
    print("  P_H . I  (unsigned sum over H):")
    for name,I in tests.items():
        r=PH(I,H)
        z = "  <-- ZERO" if not r else ""
        terms=", ".join(f"{v}*{list(k)}" for k,v in list(r.items())[:4])
        print(f"    {name:16s}: {len(r)} terms{z}   {terms}{' ...' if len(r)>4 else ''}")
    print()
