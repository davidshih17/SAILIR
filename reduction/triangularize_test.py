#!/usr/bin/env python
"""DIRECT test: are the invertible numerator symmetries SIMULTANEOUSLY TRIANGULARIZABLE?
Build a common invariant flag by finding a common eigenvector, quotienting, recursing.
Full flag exists  <=>  triangularizable  <=>  every integral CAN be reduced to a single
lex representative (+ lower).  No common eigenvector at some level => genuine residual.
Test all 8 four-prop sectors of the pentagon box."""
import sys, os, itertools
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
def nullspace(M,d):   # basis of right kernel of d x d matrix over GF(P)
    A=[r[:] for r in M]; piv={}; r=0
    for c in range(d):
        p=next((i for i in range(r,d) if A[i][c]%P),None)
        if p is None: continue
        A[r],A[p]=A[p],A[r]; iv=inv(A[r][c]); A[r]=[(x*iv)%P for x in A[r]]
        for i in range(d):
            if i!=r and A[i][c]%P:
                f=A[i][c]; A[i]=[(A[i][j]-f*A[r][j])%P for j in range(d)]
        piv[c]=r; r+=1
    free=[c for c in range(d) if c not in piv]
    basis=[]
    for fc in free:
        v=[0]*d; v[fc]=1
        for c,rr in piv.items(): v[c]=(-A[rr][fc])%P
        basis.append(v)
    return basis
def common_eigvec(gens,d):
    # candidate subspaces = list of bases; intersect eigenspaces over all gens
    spaces=[eye(d)]  # whole space (rows = basis vectors)
    for g in gens:
        ns=[]
        for sp in spaces:
            m=len(sp)
            # g restricted to sp: need sp invariant under g? build g|sp in sp-coords via least squares in GF(P)
            # Represent sp as columns; solve g*sp_col = sp_col * R  -> but sp may not be invariant.
            # Simpler: eigenvectors of g that lie in sp. For each eigenvalue lam, kernel(g-lam) ∩ sp.
            for lam in range(P):
                Mm=[[(g[i][j]-(lam if i==j else 0))%P for j in range(d)] for i in range(d)]
                if det(Mm,d)!=0: continue
                ker=nullspace(Mm,d)
                # intersect ker with sp
                inter=intersect(ker,sp,d)
                if inter: ns.append(inter)
        spaces=ns
        if not spaces: return None
    for sp in spaces:
        if sp: return sp[0]
    return None
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
def intersect(A,Bs,d):
    # intersection of span(A) and span(Bs): solve for combos. Use: v in A and in B.
    # Build matrix [A^T | -B^T]; nullspace gives combos; map to vectors.
    if not A or not Bs: return []
    cols=A+Bs; K=len(A); L=len(Bs)
    # solve sum a_i A_i = sum b_j B_j  -> [A_i ; -B_j] combos in kernel
    Mrows=[]
    for x in range(d):
        Mrows.append([A[i][x] for i in range(K)]+[(-Bs[j][x])%P for j in range(L)])
    ker=nullspace_rect(Mrows,d,K+L)
    out=[]
    for cvec in ker:
        v=[0]*d
        for i in range(K):
            for x in range(d): v[x]=(v[x]+cvec[i]*A[i][x])%P
        if any(t%P for t in v): out.append(v)
    return rref(out,d)
def nullspace_rect(M,rows,cols):
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
def triangularizable(gens,d):
    if d<=1: return True
    v=common_eigvec(gens,d)
    if v is None: return False
    # extend v to basis, change coords, recurse on (d-1) quotient block
    Bm=[v[:]]
    for e in range(d):
        cand=[1 if k==e else 0 for k in range(d)]
        if len(rref(Bm+[cand],d))>len(rref(Bm,d)): Bm.append(cand)
        if len(Bm)==d: break
    Binv=matinv(Bm,d)   # Bm rows are new basis; change of basis
    newg=[mm(mm(Bm,g,d),Binv,d) for g in gens]   # g in new basis = Bm g Bm^{-1}
    quo=[[[row[j] for j in range(1,d)] for row in gg[1:]] for gg in newg]  # lower-right (d-1) block
    return triangularizable(quo,d-1)
def matinv(M,d):
    A=[M[i][:]+[1 if j==i else 0 for j in range(d)] for i in range(d)]
    for c in range(d):
        p=next((i for i in range(c,d) if A[i][c]%P),None)
        A[c],A[p]=A[p],A[c]; iv=inv(A[c][c]); A[c]=[(x*iv)%P for x in A[c]]
        for i in range(d):
            if i!=c and A[i][c]%P:
                f=A[i][c]; A[i]=[(A[i][j]-f*A[c][j])%P for j in range(2*d)]
    return [row[d:] for row in A]

# all 8 four-prop sectors
for combo in itertools.combinations(range(ND),4):
    corner=tuple(1 if k in combo else 0 for k in range(11))
    S=combo; slots=[j for j in range(ND) if j not in S]+[8,9,10]
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
    if not nontriv:
        print(f"sector {S}: no nontrivial invertible symmetry (trivial)"); continue
    tri=triangularizable(nontriv,n)
    print(f"sector {S}: {len(nontriv)} invertible syms -> simultaneously TRIANGULARIZABLE = {tri}")
