#!/usr/bin/env python
"""Separate INVERTIBLE symmetries (the genuine group -> Maschke applies) from
non-invertible relations in the same-weight numerator action. Re-test whether the
GROUP (invertible part only) is abelian / triangularizable, per sector."""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def sec(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
def inv(a): return pow(a,P-2,P)
def det(M,d):
    M=[r[:] for r in M]; s=1
    for c in range(d):
        p=next((i for i in range(c,d) if M[i][c]%P),None)
        if p is None: return 0
        if p!=c: M[c],M[p]=M[p],M[c]; s=-s
        iv=inv(M[c][c])
        for i in range(c+1,d):
            f=(M[i][c]*iv)%P
            if f: M[i]=[(M[i][j]-f*M[c][j])%P for j in range(d)]
    dv=s%P
    for c in range(d): dv=(dv*M[c][c])%P
    return dv%P
def mm(A,Bm,d): return [[sum(A[i][k]*Bm[k][j] for k in range(d))%P for j in range(d)] for i in range(d)]

for corner in [(0,1,1,0,1,0,0,1,0,0,0),(1,1,1,1,0,0,0,0,0,0,0),(1,1,0,0,1,1,0,0,0,0,0)]:
    S=tuple(k for k in range(ND) if corner[k]>0)
    slots=[j for j in range(ND) if j not in S]+[8,9,10]
    B=[tuple(-1 if k==j else corner[k] for k in range(11)) for j in slots]
    Bi={b:i for i,b in enumerate(B)}; n=len(B); csec=sec(corner)
    def proj(t):
        M=[[0]*n for _ in range(n)]
        for j,b in enumerate(B):
            img=image_unsigned(b,*t)
            if img is None: continue
            for k,co in img.items():
                if k in Bi: M[Bi[k]][j]=co%P
        return M
    autos=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and sec(next(iter(im)))==csec)(image_unsigned(corner,*t))]
    mats=[proj(t) for t in autos]
    dets=[det(M,n) for M in mats]
    inv_mats=[M for M,dd in zip(mats,dets) if dd!=0]
    sing=sum(1 for dd in dets if dd==0)
    ident=[[1 if i==j else 0 for j in range(n)] for i in range(n)]
    nontriv=[M for M in inv_mats if M!=ident]
    ab=all(mm(a,b,n)==mm(b,a,n) for a in nontriv for b in nontriv)
    # is the invertible set closed (a group)?  build closure
    def key(M): return tuple(tuple(r) for r in M)
    G={key(ident)}; fr=[ident]; ok=True
    while fr:
        x=fr.pop()
        for g in nontriv:
            y=mm(x,g,n)
            if det(y,n)==0: ok=False
            if key(y) not in G: G.add(key(y)); fr.append(y)
        if len(G)>5000: break
    print(f"sector {S}: {len(autos)} autos  ->  {len(inv_mats)} INVERTIBLE (group), {sing} singular (relations)")
    print(f"   invertible-group abelian? {ab}    closed(all products invertible)? {ok}    |generated|={len(G)}")
    print()
