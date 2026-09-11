#!/usr/bin/env python
"""Single monomials don't vanish, but COMBINATIONS can. Measure it: for a symmetric
sector, take all ISP monomials up to degree d, symmetrize each (signed P_H), and
compute rank over GF(p). N_monomials - rank = dimension of antisymmetric combinations
that vanish under P_S. Show a concrete vanishing combination."""
import sys, os, itertools
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned, RED, P
ISP=[8,9,10]
def secbits(i): return tuple(1 if i[k]>0 else 0 for k in range(8))
def stab(corner):
    return [(M,c) for (M,c) in _transforms(corner)
            if (lambda im: im and len(im)==1 and next(iter(im))==corner)(image_unsigned(corner,M,c))]
def PH(I,H):
    acc={}
    for (M,c) in H:
        img=RED._image(I,M,c)
        if img is None: continue
        for k,v in img.items(): acc[k]=(acc.get(k,0)+v)%P
    return {k:v for k,v in acc.items() if v%P}
def inv(a): return pow(a,P-2,P)
def rank(rows,cols):
    rows=[r[:] for r in rows]; r=0
    for c in range(cols):
        piv=next((i for i in range(r,len(rows)) if rows[i][c]%P),None)
        if piv is None: continue
        rows[r],rows[piv]=rows[piv],rows[r]; iv=inv(rows[r][c])
        rows[r]=[(x*iv)%P for x in rows[r]]
        for i in range(len(rows)):
            if i!=r and rows[i][c]%P:
                f=rows[i][c]; rows[i]=[(rows[i][j]-f*rows[r][j])%P for j in range(cols)]
        r+=1
        if r==len(rows): break
    return r

for combo,dmax in [((0,2,4,5),3),((0,3,4,5),3),((1,2,4,7),3)]:
    corner=tuple(1 if k in combo else 0 for k in range(11)); H=stab(corner)
    monos=[]; 
    for deg in range(1,dmax+1):
        for cp in itertools.combinations_with_replacement(ISP,deg):
            I=list(corner)
            for s in cp: I[s]-=1
            monos.append(tuple(I))
    imgs=[PH(I,H) for I in monos]
    allk=sorted({k for im in imgs for k in im})
    idx={k:j for j,k in enumerate(allk)}
    mat=[[im.get(k,0)%P for k in allk] for im in imgs]   # rows = monomials, cols = image integrals
    rk=rank([row[:] for row in mat], len(allk))
    N=len(monos); ant = N-rk
    print(f"sector {sorted(combo)} |H|={len(H)} deg<= {dmax}: {N} monomials -> symmetric rank {rk}, "
          f"VANISHING-combination dim = {ant }")
    # exhibit one vanishing combination: find null vector of mat^T (combo of monomials -> 0)
    if ant1:=ant1 if False else ant :  # noqa
        pass
