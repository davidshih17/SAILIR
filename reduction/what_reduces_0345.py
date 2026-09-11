#!/usr/bin/env python
"""What do the symmetry relations DO to the {0,3,4,5} degree-1 numerators?
Masters at a weight level = the G-INVARIANT (fully symmetric) combinations; everything
in a nontrivial irrep (1D-nontrivial OR >=2D) is in the augmentation submodule and gets
expressed via the relations. Compute: (a) dim of invariant subspace = # same-weight
masters; (b) is the 2D irreducible block inside the invariants or not; (c) explicitly
row-reduce the actual symmetry relations (full images, incl. lower weight) to see whether
the two block integrals reduce to LOWER weight or stay."""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def sec(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
def wt(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0), tuple(-abs(x) for x in i))  # LARGER = higher on all 3 components (matches ibp_env.weight)
def inv(a): return pow(a,P-2,P)
combo=(0,3,4,5); corner=tuple(1 if k in combo else 0 for k in range(11))
slots=[j for j in range(ND) if j not in combo]+[8,9,10]
B=[tuple(-1 if k==j else corner[k] for k in range(11)) for j in slots]
Bi={b:i for i,b in enumerate(B)}; n=len(B); csec=sec(corner)
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
def eye(d): return [[1 if i==j else 0 for j in range(d)] for i in range(d)]
nontriv=[M for M in inv_g if M!=eye(n)]

# invariant subspace = intersection of ker(M_g - I) over all g
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
# stack all (M_g - I) rows; invariants = nullspace of that stacked matrix
stack=[]
for M in nontriv:
    for i in range(n):
        stack.append([(M[i][j]-(1 if i==j else 0))%P for j in range(n)])
def nullspace(rows,w):
    A=[r[:] for r in rows]; piv={}; r=0; R=len(A)
    for c in range(w):
        p=next((i for i in range(r,R) if A[i][c]%P),None)
        if p is None: continue
        A[r],A[p]=A[p],A[r]; iv=inv(A[r][c]); A[r]=[(x*iv)%P for x in A[r]]
        for i in range(R):
            if i!=r and A[i][c]%P:
                f=A[i][c]; A[i]=[(A[i][j]-f*A[r][j])%P for j in range(w)]
        piv[c]=r; r+=1
    free=[c for c in range(w) if c not in piv]; basis=[]
    for fc in free:
        v=[0]*w; v[fc]=1
        for c,rr in piv.items(): v[c]=(-A[rr][fc])%P
        basis.append(v)
    return basis
invsub=nullspace(stack,n)
print(f"sector {combo}: degree-1 same-weight numerator space dim = {n} (slots {slots})")
print(f"  G-INVARIANT subspace dim = {len(invsub)}  = number of same-weight MASTERS")
def show(v): return " + ".join(f"{c%P}*num@{slots[i]}" for i,c in enumerate(v) if c%P) or "0"
for v in invsub: print("     master (symmetric combo):", show(v))
print(f"  => the other {n-len(invsub)} dims (incl. the 2D irreducible block) are NOT invariant")

# explicitly reduce one block integral via the real relations (full images w/ lower weight)
print("\n--- explicit reduction of num@1 and num@9 via symmetry relations ---")
for seedslot in (slots[0], 9):
    u=tuple(-1 if k==seedslot else corner[k] for k in range(11))
    # collect relations u - sigma_g(u) = 0 (full images incl lower weight), plus for all its images
    rels=[]; seen=set()
    work=[u]
    while work:
        x=work.pop()
        if x in seen: continue
        seen.add(x)
        for t in autos:
            img=image_unsigned(x,*t)
            if img is None: continue
            rel={x:1}
            for k,c in img.items(): rel[k]=(rel.get(k,0)-c)%P
            rel={k:c for k,c in rel.items() if c%P}
            if rel: rels.append(rel)
            for k in img:  # keep same-weight partners in the system
                if wt(k)[:2]==wt(u)[:2] and k not in seen: work.append(k)
    allints=sorted({k for r in rels for k in r}, key=wt, reverse=True)  # high weight first
    col={ii:j for j,ii in enumerate(allints)}
    mat=[[r.get(ii,0)%P for ii in allints] for r in rels]
    red=rref(mat,len(allints))
    # which same-weight integrals are pivots (=> expressed in terms of lower)?
    reduced=[]
    for row in red:
        pc=next((j for j in range(len(allints)) if row[j]%P),None)
        if pc is None: continue
        piv_int=allints[pc]
        if wt(piv_int)[:2]==wt(u)[:2]:
            tail=[allints[j] for j in range(pc+1,len(allints)) if row[j]%P]
            lower=all(wt(t2)<wt(piv_int) for t2 in tail)
            reduced.append((piv_int,len(tail),lower))
    print(f"  seed num@{seedslot} = {u}, weight {wt(u)[:2]}:")
    print(f"    relations reduce {len(reduced)} same-weight integral(s); each to strictly-lower weight? {all(l for _,_,l in reduced)}")
    for pi,nt,lo in reduced[:3]:
        print(f"      {pi}  -> {'LOWER-weight combo' if lo else 'same-or-mixed'} ({nt} tail terms)")
