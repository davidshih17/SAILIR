#!/usr/bin/env python
"""Independently verify sector {0,3,4,5} is genuinely non-triangularizable:
(1) NO common eigenvector for the invertible generators (direct check);
(2) smallest invariant block is absolutely IRREDUCIBLE (Burnside dim = d^2);
(3) the projective group order is COPRIME to 1009 (=> semisimple, real 2D irrep,
    not a characteristic-p / affine-shift artifact).
Also sanity-check the triangularizer on a hand-built rotation (should be False) and a
triangular pair (should be True)."""
import sys, os, math
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def sec(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
def inv(a): return pow(a,P-2,P)
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
def nullspace(M,d):
    A=[r[:] for r in M]; piv={}; r=0
    for c in range(d):
        p=next((i for i in range(r,d) if A[i][c]%P),None)
        if p is None: continue
        A[r],A[p]=A[p],A[r]; iv=inv(A[r][c]); A[r]=[(x*iv)%P for x in A[r]]
        for i in range(d):
            if i!=r and A[i][c]%P:
                f=A[i][c]; A[i]=[(A[i][j]-f*A[r][j])%P for j in range(d)]
        piv[c]=r; r+=1
    free=[c for c in range(d) if c not in piv]; basis=[]
    for fc in free:
        v=[0]*d; v[fc]=1
        for c,rr in piv.items(): v[c]=(-A[rr][fc])%P
        basis.append(v)
    return basis
def rref(rows,w):
    rows=[r[:] for r in rows if any(x%P for x in r)]; piv=[]; r=0
    for c in range(w):
        p=next((i for i in range(r,len(rows)) if rows[i][c]%P),None)
        if p is None: continue
        rows[r],rows[p]=rows[p],rows[r]; iv=inv(rows[r][c]); rows[r]=[(x*iv)%P for x in rows[r]]
        for i in range(len(rows)):
            if i!=r and rows[i][c]%P:
                f=rows[i][c]; rows[i]=[(rows[i][j]-f*rows[r][j])%P for j in range(w)]
        piv.append(c); r+=1
        if r==len(rows): break
    return rows[:r]
def dimspan(vs,w): return len(rref(vs,w))
def eigvals(M,d):
    return [lam for lam in range(P) if det([[(M[i][j]-(lam if i==j else 0))%P for j in range(d)] for i in range(d)],d)==0]

# ---- common eigenvector across a set of matrices (direct: intersect eigenspaces) ----
def common_eigvec(gens,d):
    spaces=[eye(d)]
    for g in gens:
        newsp=[]
        evs=eigvals(g,d)
        for sp in spaces:
            for lam in evs:
                Mm=[[(g[i][j]-(lam if i==j else 0))%P for j in range(d)] for i in range(d)]
                ker=nullspace(Mm,d)
                inter=intersect(ker,sp,d)
                if inter: newsp.append(inter)
        spaces=newsp
        if not spaces: return None
    return spaces[0][0] if spaces and spaces[0] else None
def intersect(A,Bs,d):
    if not A or not Bs: return []
    K,L=len(A),len(Bs); Mrows=[]
    for x in range(d): Mrows.append([A[i][x] for i in range(K)]+[(-Bs[j][x])%P for j in range(L)])
    ker=nullrect(Mrows,d,K+L); out=[]
    for cvec in ker:
        v=[0]*d
        for i in range(K):
            for x in range(d): v[x]=(v[x]+cvec[i]*A[i][x])%P
        if any(t%P for t in v): out.append(v)
    return rref(out,d)
def nullrect(M,rows,cols):
    A=[r[:] for r in M]; piv={}; r=0
    for c in range(cols):
        p=next((i for i in range(r,rows) if A[i][c]%P),None)
        if p is None: continue
        A[r],A[p]=A[p],A[r]; iv=inv(A[r][c]); A[r]=[(x*iv)%P for x in A[r]]
        for i in range(rows):
            if i!=r and A[i][c]%P:
                f=A[i][c]; A[i]=[(A[i][j]-f*A[r][j])%P for j in range(cols)]
        piv[c]=r; r+=1
    free=[c for c in range(cols) if c not in piv]; basis=[]
    for fc in free:
        v=[0]*cols; v[fc]=1
        for c,rr in piv.items(): v[c]=(-A[rr][fc])%P
        basis.append(v)
    return basis

# sanity checks on the common_eigvec primitive
rot=[[0,P-1],[1,0]]          # order-4 rotation, no eigenvector over... has eigenvalues? x^2+1=0
tri=[[1,1],[0,2]]
print("sanity: common_eigvec(rotation [[0,-1],[1,0]]) =", common_eigvec([rot],2), "(None if no GF(P) eigenvector)")
print("sanity: common_eigvec(triangular [[1,1],[0,2]]) =", common_eigvec([tri],2), "(should find one)")

# ---- {0,3,4,5} ----
combo=(0,3,4,5); corner=tuple(1 if k in combo else 0 for k in range(11))
slots=[j for j in range(ND) if j not in combo]+[8,9,10]
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
inv_g=[M for M in (proj(t) for t in autos) if det(M,n)!=0]
nontriv=[M for M in inv_g if M!=eye(n)]
print(f"\n{combo}: {len(nontriv)} nontrivial invertible syms, numerator dim {n}")
ce=common_eigvec(nontriv,n)
print(f"  common eigenvector across ALL generators: {ce}")
print(f"  => {'HAS common eigvec (triangularizable first step)' if ce else 'NO common eigenvector -> NOT triangularizable (genuine obstruction)'}")

# projective group order + coprimality to 1009
def normalize(M,d):
    for i in range(d):
        for j in range(d):
            if M[i][j]%P:
                iv=inv(M[i][j]); return tuple(tuple((M[a][b]*iv)%P for b in range(d)) for a in range(d))
G={normalize(eye(n),n)}; fr=[eye(n)]
while fr:
    x=fr.pop()
    for g in nontriv:
        y=mm(x,g,n); ny=normalize(y,n)
        if ny not in G: G.add(ny); fr.append([list(r) for r in ny])
    if len(G)>4000: break
print(f"  projective group order = {len(G)}   1009 divides it? {len(G)%1009==0}")
print(f"  (coprime to 1009 => semisimple => the non-triangularizability is a REAL 2D+ irrep)")

# smallest invariant block + Burnside
def mvn(M,v): return [sum(M[i][k]*v[k] for k in range(n))%P for i in range(n)]
def inv_span(seed):
    basis=[seed[:]]; frl=[seed[:]]
    while frl:
        v=frl.pop()
        for g in nontriv:
            w=mvn(g,v)
            if dimspan(basis+[w],n)>dimspan(basis,n): basis.append(w); frl.append(w)
        if len(basis)>=n: break
    return rref(basis,n)
seeds=[[1 if k==j else 0 for k in range(n)] for j in range(n)]
for i in range(n):
    for j in range(i+1,n):
        for s in (1,P-1): v=[0]*n; v[i]=1; v[j]=s; seeds.append(v)
blocks=[W for W in (inv_span(sd) for sd in seeds) if len(W)>=2]
W=min(blocks,key=len); d=len(W); red=rref(W,n)
piv=[next(j for j in range(n) if row[j]%P) for row in red]
def coords(vn):
    c=[0]*d; tmp=vn[:]
    for b,pc in enumerate(piv):
        c[b]=tmp[pc]%P; tmp=[(tmp[k]-c[b]*red[b][k])%P for k in range(n)]
    return None if any(x%P for x in tmp) else c
def restr(g):
    R=[[0]*d for _ in range(d)]
    for a in range(d):
        gc=coords(mvn(g,red[a]))
        if gc is None: return None
        for b in range(d): R[b][a]=gc[b]
    return R
rest=[R for R in (restr(g) for g in nontriv) if R is not None]
def flat(M): return tuple(M[i][j] for i in range(d) for j in range(d))
words=[eye(d)]+rest; fs=[flat(w) for w in words]; ch=True
while ch:
    ch=False; cur=[list(x) for x in fs]
    for x in list(words):
        for g in rest:
            pr=mm(x,g,d)
            if dimspan(cur+[flat(pr)],d*d)>dimspan(cur,d*d):
                words.append(pr); fs.append(flat(pr)); cur.append(list(flat(pr))); ch=True
algdim=dimspan(fs,d*d)
print(f"  smallest invariant block dim d={d}, Burnside algebra dim={algdim} (d^2={d*d}) -> {'ABSOLUTELY IRREDUCIBLE' if algdim==d*d and d>=2 else 'reducible'}")
def show(v): return " + ".join(f"{c%P}*num@{slots[i]}" for i,c in enumerate(v) if c%P)
for row in red: print("     block basis:",show(row))
