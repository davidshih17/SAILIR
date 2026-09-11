#!/usr/bin/env python
"""Build the numerator (ISP) representation of a sector's symmetry group and test
whether it is triangularizable. For a finite group in char not dividing |G| the rep
is semisimple, so: triangularizable <=> matrices commute (abelian image) <=> no
genuine >=2D irrep. Commuting => the user's total-order+shift can reach a signed
permutation. Non-commuting => an irreducible >=2D residual survives any term order."""
import sys, os, itertools
BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from symmetry_engine import N
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def sector(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))

def build_rep(corner):
    S = tuple(k for k in range(ND) if corner[k]>0)          # present props
    num_slots = [j for j in range(ND) if j not in S] + [8,9,10]
    # degree-1 numerator basis over the corner
    B=[]
    for j in num_slots:
        t=list(corner); t[j]=-1; B.append(tuple(t))
    Bindex={b:i for i,b in enumerate(B)}
    print(f"corner {corner}  sector props {S}  numerator basis dim = {len(B)} (slots {num_slots})")
    mats=[]; nmono=0; ntransf=0
    for (M,c) in _transforms(corner):
        imgZ = image_unsigned(corner, M, c)
        if imgZ is None: continue
        # automorphism of the sector: corner maps to a single same-sector corner
        if not (len(imgZ)==1 and sector(next(iter(imgZ)))==tuple(1 if corner[k]>0 else 0 for k in range(ND))):
            continue
        ntransf+=1
        Mrep=[[0]*len(B) for _ in range(len(B))]
        rowcounts=[]
        for jcol,b in enumerate(B):
            img=image_unsigned(b,M,c)
            if img is None: continue
            nz=0
            for k,co in img.items():
                if k in Bindex:                              # same-weight numerator term
                    Mrep[Bindex[k]][jcol]=co%P; nz+=1
            rowcounts.append(nz)
        # monomial? (<=1 nonzero per column)
        if max(rowcounts or [0])<=1: nmono+=1
        mats.append(Mrep)
    print(f"sector automorphisms found: {ntransf}   (of which act monomially on numerators: {nmono})")
    return mats,len(B)

def matmul(A,Bm,n):
    return [[sum(A[i][k]*Bm[k][j] for k in range(n))%P for j in range(n)] for i in range(n)]
def commute(A,Bm,n):
    return matmul(A,Bm,n)==matmul(Bm,A,n)

# try a few sectors with (hopefully) nontrivial automorphisms
for corner in [(0,1,1,0,1,0,0,1,0,0,0),(1,1,1,1,0,0,0,0,0,0,0),(1,1,1,0,1,0,0,0,0,0,0),
               (1,1,0,0,1,1,0,0,0,0,0),(0,1,1,1,1,0,0,0,0,0,0)]:
    print("="*60)
    mats,n=build_rep(corner)
    if len(mats)<2:
        print("  <2 automorphism matrices -> rep trivial/too small here\n"); continue
    # nontrivial (non-identity) matrices
    ident=[[1 if i==j else 0 for j in range(n)] for i in range(n)]
    nontriv=[Mrep for Mrep in mats if Mrep!=ident]
    allcommute=all(commute(nontriv[a],nontriv[b],n) for a in range(len(nontriv)) for b in range(a+1,len(nontriv)))
    print(f"  nontrivial matrices: {len(nontriv)}")
    print(f"  ALL PAIRS COMMUTE (abelian image => triangularizable): {allcommute}")
    print(f"  => {'NO genuine >=2D irrep: user total-order+shift CAN reach signed permutation' if allcommute else 'NON-abelian => genuine >=2D irrep => irreducible residual mixing'}\n")
