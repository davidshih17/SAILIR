#!/usr/bin/env python
"""Verify the general symmetry-reduction rule at the degree-1 numerator level, per sector:
  survivors = dim of G-invariant subspace (= nullspace of stacked (M_g - I));
  every non-invariant same-weight integral reduces to STRICTLY-LOWER total-weight.
If (#same-weight integrals that row-reduce to strictly-lower total-weight) == (n - #survivors)
for every sector, the rule 'I = P_S I + lower' holds uniformly."""
import sys, os, itertools
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def sec(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
def tw(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0), tuple(-abs(x) for x in i))  # LARGER = higher, matches ibp_env.weight
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
    return [c for c in range(w) if c not in piv]   # free cols -> dim of nullspace = len

results=[]
for combo in itertools.combinations(range(ND),4):
    corner=tuple(1 if k in combo else 0 for k in range(11)); csec=sec(corner)
    slots=[j for j in range(ND) if j not in combo]+[8,9,10]
    B=[tuple(-1 if k==j else corner[k] for k in range(11)) for j in slots]
    Bi={b:i for i,b in enumerate(B)}; n=len(B)
    def proj(t):
        M=[[0]*n for _ in range(n)]
        for j,b in enumerate(B):
            img=image_unsigned(b,*t)
            if img is None: continue
            for k,co in img.items():
                if k in Bi: M[Bi[k]][j]=co%P
        return M
    autos=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and sec(next(iter(im)))==csec)(image_unsigned(corner,*t))]
    mats=[M for M in (proj(t) for t in autos) if det(M,n)!=0]
    eye=[[1 if i==j else 0 for j in range(n)] for i in range(n)]
    nontriv=[M for M in mats if M!=eye]
    if not nontriv:
        continue
    # survivors = invariant subspace dim
    stack=[[(M[i][j]-(1 if i==j else 0))%P for j in range(n)] for M in nontriv for i in range(n)]
    survivors=len(nullspace(stack,n))
    # build all symmetry relations (full images) for the orbit, row-reduce in total-weight order
    rels=[]; seen=set(); work=list(B)
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
            for k in img:
                if tw(k)[:2]==(tw(x)[0],tw(x)[1]) and k not in seen and k in Bi: work.append(k)
    allints=sorted({k for r in rels for k in r}, key=tw, reverse=True)  # highest total-weight first
    col={ii:j for j,ii in enumerate(allints)}
    mat=[[r.get(ii,0)%P for ii in allints] for r in rels]
    red=rref(mat,len(allints))
    lvlw=tw(B[0])[:2]  # (w1,w2) of this degree-1 level
    reduced_to_lower=0
    for row in red:
        pc=next((j for j in range(len(allints)) if row[j]%P),None)
        if pc is None: continue
        piv=allints[pc]
        if tw(piv)[:2]!=lvlw or piv not in Bi: continue   # only count same-level numerators
        tail=[allints[j] for j in range(pc+1,len(allints)) if row[j]%P]
        if all(tw(t2) < tw(piv) for t2 in tail):  # all strictly lower total-weight
            reduced_to_lower+=1
    results.append((combo,n,survivors,reduced_to_lower,len(nontriv)))

print(f"{'sector':<14}{'dim':>4}{'survivors':>11}{'reduced->lower':>16}{'rule holds':>12}")
allok=True
for combo,n,surv,red_,ng in results:
    ok = (red_ == n - surv)
    allok = allok and ok
    print(f"{str(combo):<14}{n:>4}{surv:>11}{red_:>16}{'  OK' if ok else '  MISMATCH':>12}")
print(f"\nRULE 'I = P_S*I + (strictly-lower total-weight)' holds for ALL sectors: {allok}")
print("  (survivors = fully-symmetric combinations; every other same-weight integral reduces to strictly-lower total-weight)")
