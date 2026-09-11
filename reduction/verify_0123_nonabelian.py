#!/usr/bin/env python
"""Rigorously characterize the INVERTIBLE symmetry group of sector {0,1,2,3} on the
numerator space: element orders, an explicit non-commuting pair (and whether they even
commute projectively), and the smallest irreducible block via Burnside. This decides
whether the non-abelian verdict is real or a scalar/relation artifact."""
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
def order(M,d,cap=5000):
    X=M; k=1
    while X!=eye(d):
        X=mm(X,M,d); k+=1
        if k>cap: return None
    return k
def is_scalar_mult(A,Bm,d):
    # is A == lambda*B for some scalar? return lambda or None
    lam=None
    for i in range(d):
        for j in range(d):
            if Bm[i][j]%P:
                cand=(A[i][j]*inv(Bm[i][j]))%P
                if lam is None: lam=cand
                elif lam!=cand: return None
            elif A[i][j]%P: return None
    return lam

autos=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and sec(next(iter(im)))==csec)(image_unsigned(corner,*t))]
mats=[proj(t) for t in autos]
inv_g=[M for M in mats if det(M,n)!=0]
I=eye(n); nontriv=[M for M in inv_g if M!=I]
print(f"sector {S}: {len(inv_g)} invertible symmetries; element orders (non-identity):")
for M in nontriv: print(f"   order = {order(M,n)}")

# explicit non-commuting pair + projective check
pair=None
for a in range(len(nontriv)):
    for b in range(a+1,len(nontriv)):
        AB=mm(nontriv[a],nontriv[b],n); BA=mm(nontriv[b],nontriv[a],n)
        if AB!=BA:
            pair=(a,b, is_scalar_mult(AB,BA,n)); break
    if pair: break
print(f"\nnon-commuting pair among invertible symmetries: {(pair[0],pair[1]) if pair else None}")
if pair:
    lam=pair[2]
    print(f"  AB == lambda*BA ?  {'yes, lambda='+str(lam)+' (projectively commute)' if lam is not None else 'NO - not even projectively (genuine >=2D irrep)'}")

# smallest dim>=2 invariant block under invertible group, Burnside test
def rref(rows,w):
    rows=[r[:] for r in rows]; piv=[]; r=0
    for c in range(w):
        p=next((i for i in range(r,len(rows)) if rows[i][c]%P),None)
        if p is None: continue
        rows[r],rows[p]=rows[p],rows[r]; iv=inv(rows[r][c]); rows[r]=[(x*iv)%P for x in rows[r]]
        for i in range(len(rows)):
            if i!=r and rows[i][c]%P:
                f=rows[i][c]; rows[i]=[(rows[i][j]-f*rows[r][j])%P for j in range(w)]
        piv.append(c); r+=1
        if r==len(rows): break
    return rows[:r],piv
def dimspan(vs,w): return len(rref(vs,w)[0])
def mvn(M,v): return [sum(M[i][k]*v[k] for k in range(n))%P for i in range(n)]
def inv_span(seed):
    basis=[seed[:]]; fr=[seed[:]]
    while fr:
        v=fr.pop()
        for g in nontriv:
            w=mvn(g,v)
            if dimspan(basis+[w],n)>dimspan(basis,n): basis.append(w); fr.append(w)
        if len(basis)>=n: break
    return rref(basis,n)[0]
seeds=[[1 if k==j else 0 for k in range(n)] for j in range(n)]
for i in range(n):
    for j in range(i+1,n):
        for s in (1,P-1): v=[0]*n; v[i]=1; v[j]=s; seeds.append(v)
cand=[inv_span(sd) for sd in seeds]
ge2=[W for W in cand if len(W)>=2]
W=min(ge2,key=len)
d=len(W); red,piv=rref(W,n)
def coords(vn):
    c=[0]*d; tmp=vn[:]
    for b,pc in enumerate(piv):
        c[b]=tmp[pc]%P; tmp=[(tmp[k]-c[b]*red[b][k])%P for k in range(n)]
    return None if any(x%P for x in tmp) else c
def restrict(g):
    R=[[0]*d for _ in range(d)]
    for a in range(d):
        gc=coords(mvn(g,red[a]))
        if gc is None: return None
        for b in range(d): R[b][a]=gc[b]
    return R
rest=[R for R in (restrict(g) for g in nontriv) if R is not None]
# Burnside algebra dim
def flat(M): return tuple(M[i][j] for i in range(d) for j in range(d))
words=[eye(d)]+rest; fs=[flat(w) for w in words]; changed=True
while changed:
    changed=False; cur=[list(x) for x in fs]
    for x in list(words):
        for g in rest:
            pr=mm(x,g,d)
            if dimspan(cur+[flat(pr)],d*d)>dimspan(cur,d*d):
                words.append(pr); fs.append(flat(pr)); cur.append(list(flat(pr))); changed=True
algdim=dimspan(fs,d*d)
print(f"\nsmallest invariant block under invertible group: dim {d}")
def show(v): return " + ".join(f"{c%P}*num@{slots[i]}" for i,c in enumerate(v) if c%P)
for row in red: print("    basis:",show(row))
print(f"  Burnside algebra dim = {algdim} (d^2={d*d})  ->  {'ABSOLUTELY IRREDUCIBLE' if algdim==d*d and d>=2 else 'reducible'}")
for idx,R in enumerate(rest):
    print(f"  gen{idx}|block ="); 
    for r in R: print("       ",r)
