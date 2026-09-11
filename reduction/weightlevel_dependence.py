#!/usr/bin/env python
"""Is the reduction recipe sector-level or weight-level? Build sector {0,3,4,5}'s
numerator rep at numerator-degree 1 and degree 2 over the corner, and report the
spectrum of invariant-subspace dims + the smallest irreducible block for each.
Same group both times (sector automorphisms), but the block structure should DIFFER
by degree -> recipe is per (sector, weight-level), not per individual integral."""
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
def eye(d): return [[1 if i==j else 0 for j in range(d)] for i in range(d)]
def mm(A,Bm,d): return [[sum(A[i][k]*Bm[k][j] for k in range(d))%P for j in range(d)] for i in range(d)]
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

combo=(0,3,4,5); corner=tuple(1 if k in combo else 0 for k in range(11))
ISP=[j for j in range(ND) if j not in combo]+[8,9,10]   # 7 numerator directions
csec=sec(corner)
autos=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and sec(next(iter(im)))==csec)(image_unsigned(corner,*t))]

def build_basis(deg):
    # numerator monomials of total degree `deg` over ISP slots, as integrals over corner
    B=[]
    for combo_slots in itertools.combinations_with_replacement(ISP,deg):
        t=list(corner)
        for s in combo_slots: t[s]-=1
        B.append(tuple(t))
    return B

def analyze(deg):
    B=build_basis(deg); Bi={b:i for i,b in enumerate(B)}; n=len(B)
    def proj(tr):
        M=[[0]*n for _ in range(n)]
        for j,b in enumerate(B):
            img=image_unsigned(b,*tr)
            if img is None: continue
            for k,co in img.items():
                if k in Bi: M[Bi[k]][j]=co%P
        return M
    inv_g=[M for M in (proj(t) for t in autos) if det(M,n)!=0]
    nontriv=[M for M in inv_g if M!=eye(n)]
    def mvn(M,v): return [sum(M[i][k]*v[k] for k in range(n))%P for i in range(n)]
    def inv_span(seed):
        basis=[seed[:]]; fr=[seed[:]]
        while fr:
            v=fr.pop()
            for g in nontriv:
                w=mvn(g,v)
                if dimspan(basis+[w],n)>dimspan(basis,n): basis.append(w); fr.append(w)
            if len(basis)>=n: break
        return rref(basis,n)
    seeds=[[1 if k==j else 0 for k in range(n)] for j in range(n)]
    spectrum=sorted({len(inv_span(sd)) for sd in seeds})
    # smallest >=2 invariant block + Burnside
    blks=[W for W in (inv_span(sd) for sd in seeds) if len(W)>=2]
    info=""
    if blks:
        W=min(blks,key=len); d=len(W); red=rref(W,n)
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
        info=f"smallest >=2 block dim={d}, Burnside={algdim} ({'IRREDUCIBLE' if algdim==d*d else 'reducible'})"
    return n,len(nontriv),spectrum,info

for deg in (1,2):
    n,ng,spec,info=analyze(deg)
    print(f"numerator degree {deg}: rep dim={n}, {ng} invertible syms (SAME group)")
    print(f"   invariant-subspace dim spectrum from single-monomial seeds: {spec}")
    print(f"   {info}\n")
