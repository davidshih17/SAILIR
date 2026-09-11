#!/usr/bin/env python
"""Scan every symmetric sector's numerator integrals (ISP monomials up to degree 4)
and count how many give signed P_H.I == 0. Answers: do single integrals vanish under
P_S, and how often?"""
import sys, os, itertools
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned, RED, P

def secbits(i): return tuple(1 if i[k]>0 else 0 for k in range(8))
def stab(corner):
    return [(M,c) for (M,c) in _transforms(corner)
            if (lambda im: im and len(im)==1 and next(iter(im))==corner)(image_unsigned(corner,M,c))]
def PH_signed(I,H):
    acc={}
    for (M,c) in H:
        img=RED._image(I,M,c)
        if img is None: continue
        for k,v in img.items(): acc[k]=(acc.get(k,0)+v)%P
    return {k:v for k,v in acc.items() if v%P}

ISP=[8,9,10]  # D9,D10,D11 positions
tot_int=tot_zero=0; examples=[]; sectors_with_sym=0
for mask in range(256):
    combo=[k for k in range(8) if mask>>k&1]
    corner=tuple(1 if k in combo else 0 for k in range(11))
    H=stab(corner)
    if len(H)<=1: continue
    sectors_with_sym+=1
    # enumerate ISP monomials up to total degree 4 (also allow one dot on a present prop? keep to ISPs for clarity)
    for deg in range(1,5):
        for combo_pow in itertools.combinations_with_replacement(ISP, deg):
            I=list(corner)
            for s in combo_pow: I[s]-=1
            I=tuple(I)
            tot_int+=1
            r=PH_signed(I,H)
            if not r:
                tot_zero+=1
                if len(examples)<8: examples.append((sorted(combo),I))
print(f"symmetric sectors scanned: {sectors_with_sym}")
print(f"ISP-numerator integrals tested (deg 1-4): {tot_int}")
print(f"integrals with signed P_H.I == 0 : {tot_zero}  ({100*tot_zero/max(1,tot_int):.2f}%)")
for combo,I in examples[:8]:
    print(f"   ZERO: sector {combo}  I={list(I)}")
