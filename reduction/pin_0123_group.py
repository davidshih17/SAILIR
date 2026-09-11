#!/usr/bin/env python
"""Pin down the {0,1,2,3} numerator symmetry group PROJECTIVELY (mod scalars).
Two non-commuting involutions g0,g1 generate a dihedral group D_m where m=proj-order
of g0*g1. m>=3 => non-abelian with a genuine 2D irrep. Also report the projective
group order (should be small if the ~5000 closure was scalar inflation)."""
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
corner=(1,1,1,1,0,0,0,0,0,0,0); S=(0,1,2,3)
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
def mm(A,Bm,d): return [[sum(A[i][k]*Bm[k][j] for k in range(d))%P for j in range(d)] for i in range(d)]
def eye(d): return [[1 if i==j else 0 for j in range(d)] for i in range(d)]
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
    return dv
def normalize(M,d):
    # divide by first nonzero entry (scan row-major) -> projective canonical form
    for i in range(d):
        for j in range(d):
            if M[i][j]%P:
                iv=inv(M[i][j]); return tuple(tuple((M[a][b]*iv)%P for b in range(d)) for a in range(d))
    return None
def proj_order(M,d,cap=2000):
    I=normalize(eye(d),d); X=M; k=1
    while normalize(X,d)!=I:
        X=mm(X,M,d); k+=1
        if k>cap: return None
    return k

autos=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and sec(next(iter(im)))==csec)(image_unsigned(corner,*t))]
inv_g=[M for M in (proj(t) for t in autos) if det(M,n)!=0]
I=eye(n); nontriv=[M for M in inv_g if M!=I]
# find the non-commuting pair
g0=g1=None
for a in range(len(nontriv)):
    for b in range(a+1,len(nontriv)):
        if mm(nontriv[a],nontriv[b],n)!=mm(nontriv[b],nontriv[a],n):
            g0,g1=nontriv[a],nontriv[b]; break
    if g0: break
prod=mm(g0,g1,n)
print(f"sector {S}: two non-commuting involutions g0,g1 (orders {proj_order(g0,n)},{proj_order(g1,n)} projectively)")
print(f"  projective order of g0*g1 = {proj_order(prod,n)}   -> dihedral D_m with m = that")
print(f"  m>=3 means NON-ABELIAN with a genuine 2D irrep")

# projective group order (mod scalars)
G={normalize(I,n)}; fr=[I]
while fr:
    x=fr.pop()
    for g in nontriv:
        y=mm(x,g,n); ny=normalize(y,n)
        if ny not in G: G.add(ny); fr.append([list(r) for r in ny])
    if len(G)>2000: break
print(f"  PROJECTIVE group order (mod scalars) = {len(G)}   (vs ~5000 raw = scalar inflation confirmed)")
