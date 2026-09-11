#!/usr/bin/env python
"""Design-1 precompute/lazy budget: (1) how many sub-sectors are symmetric,
(2) at each numerator level, reducible-integrals-that-get-a-rule vs survivors-that-get-a-worker,
(3) time to GENERATE one rewrite rule (image over group + row-reduce)."""
import sys, os, itertools, time
from collections import defaultdict
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def sec(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
def inv(a): return pow(a,P-2,P)
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
def nullity(rows,w):
    A=[r[:] for r in rows]; piv=set(); r=0; R=len(A)
    for c in range(w):
        p=next((i for i in range(r,R) if A[i][c]%P),None)
        if p is None: continue
        A[r],A[p]=A[p],A[r]; ivv=inv(A[r][c]); A[r]=[(x*ivv)%P for x in A[r]]
        for i in range(R):
            if i!=r and A[i][c]%P:
                f=A[i][c]; A[i]=[(A[i][j]-f*A[r][j])%P for j in range(w)]
        piv.add(c); r+=1
    return w-len(piv)

def sym_autos(corner):
    csec=sec(corner)
    return [t for t in _transforms(corner)
            if (lambda im: im and len(im)==1 and sec(next(iter(im)))==csec)(image_unsigned(corner,*t))]

def level_basis(corner, combo, s):
    isp=[j for j in range(ND) if j not in combo]+[8,9,10]
    B=[]
    for mono in itertools.combinations_with_replacement(isp, s):
        t=list(corner)
        for j in mono: t[j]-=1
        B.append(tuple(t))
    return B

def survivors_reducible(corner, combo, s):
    B=level_basis(corner, combo, s); Bi={b:i for i,b in enumerate(B)}; n=len(B)
    autos=sym_autos(corner)
    inv_g=[]
    for t in autos:
        M=[[0]*n for _ in range(n)]
        ok=True
        for j,b in enumerate(B):
            img=image_unsigned(b,*t)
            if img is None: continue
            for k,co in img.items():
                if k in Bi: M[Bi[k]][j]=co%P
        if det(M,n)!=0: inv_g.append(M)
    if not inv_g: return n,0,n
    stack=[[(M[i][j]-(1 if i==j else 0))%P for j in range(n)] for M in inv_g if M!=eye(n) for i in range(n)]
    surv=nullity(stack,n) if stack else n
    return n, surv, n-surv

# (1) which sub-sectors are symmetric
sym_sectors=defaultdict(list)   # prop_count -> [combos]
for pcount in range(2,7):
    for combo in itertools.combinations(range(ND),pcount):
        corner=tuple(1 if k in combo else 0 for k in range(11))
        a=sym_autos(corner)
        nt=[t for t in a if image_unsigned(corner,*t)!={corner:1}]
        # invertible & nontrivial?
        if len(a)>1:
            sym_sectors[pcount].append(combo)
print("=== symmetric sub-sectors by propagator count ===")
tot_sym=0
for pc in sorted(sym_sectors):
    n=len(sym_sectors[pc]); tot_sym+=n
    print(f"  {pc}-prop sectors with nontrivial symmetry: {n} / {len(list(itertools.combinations(range(ND),pc)))}")
print(f"  TOTAL symmetric sub-sectors: {tot_sym}\n")

# (2) reducible vs survivor at s=1, s=2 (corner denominators)
print("=== integrals-with-rule (reducible) vs survivors, summed over symmetric sectors ===")
for s in (1,2):
    T=Rd=Sv=0
    for pc in sym_sectors:
        for combo in sym_sectors[pc]:
            corner=tuple(1 if k in combo else 0 for k in range(11))
            n,surv,red=survivors_reducible(corner,combo,s)
            T+=n; Sv+=surv; Rd+=red
    print(f"  s={s}: total={T}  reducible(get rule)={Rd} ({100*Rd/max(T,1):.0f}%)  survivors(get worker)={Sv} ({100*Sv/max(T,1):.0f}%)")

# (3) time to GENERATE one rewrite rule
def rref(rows,w):
    rows=[r[:] for r in rows if any(x%P for x in r)]; piv=[]; r=0
    for c in range(w):
        p=next((i for i in range(r,len(rows)) if rows[i][c]%P),None)
        if p is None: continue
        rows[r],rows[p]=rows[p],rows[r]; ivv=inv(rows[r][c]); rows[r]=[(x*ivv)%P for x in rows[r]]
        for i in range(len(rows)):
            if i!=r and rows[i][c]%P:
                f=rows[i][c]; rows[i]=[(rows[i][j]-f*rows[r][j])%P for j in range(w)]
        piv.append(c); r+=1
        if r==len(rows): break
    return rows[:r]
def gen_rule(I, combo):
    corner=tuple(1 if k in combo else 0 for k in range(11))
    autos=sym_autos(corner)
    rels=[]; ints=set([I])
    for t in autos:
        img=image_unsigned(I,*t)
        if img is None: continue
        rel={I:1}
        for k,c in img.items(): rel[k]=(rel.get(k,0)-c)%P; ints.add(k)
        rel={k:c for k,c in rel.items() if c%P}
        if rel: rels.append(rel)
    allints=sorted(ints, key=lambda x:(sum(v for v in x if v>0),sum(-v for v in x if v<0),tuple(abs(v) for v in x)), reverse=True)
    mat=[[r.get(ii,0)%P for ii in allints] for r in rels]
    return rref(mat,len(allints))
# pick a symmetric 4-prop sector, time rule gen over its s=1 numerators
combo=next(iter(sym_sectors[4])); corner=tuple(1 if k in combo else 0 for k in range(11))
isp=[j for j in range(ND) if j not in combo]+[8,9,10]
srcs=[tuple(-1 if k==j else corner[k] for k in range(11)) for j in isp]
t0=time.perf_counter()
REP=200
for _ in range(REP):
    for I in srcs: gen_rule(I,combo)
dt=(time.perf_counter()-t0)/(REP*len(srcs))
print(f"\n=== per-rule generation cost (sector {combo}, s=1) ===")
print(f"  {dt*1e6:.1f} microseconds per rule  ({1/dt:.0f} rules/sec)")
