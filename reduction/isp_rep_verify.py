#!/usr/bin/env python
"""Verify the same-weight (associated-graded) projection is a genuine group rep:
for the actual symmetry transforms, check  M(sigma)*M(tau)  ==  proj[ sigma( tau(B) ) ].
If the homomorphism holds, non-commutativity M(s)M(t)!=M(t)M(s) is a REAL property of
the symmetry group (a genuine >=2D irrep), not a projection artifact. Focus: the
non-abelian sector {0,1,2,3}."""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def sec(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))

corner=(1,1,1,1,0,0,0,0,0,0,0)                       # sector {0,1,2,3}, non-abelian above
S=tuple(k for k in range(ND) if corner[k]>0)
slots=[j for j in range(ND) if j not in S]+[8,9,10]
B=[tuple(-1 if k==j else corner[k] for k in range(11)) for j in slots]
Bi={b:i for i,b in enumerate(B)}; n=len(B)
csec=sec(corner)

def proj_action(transf):
    """matrix of one transform on the numerator basis (same-weight projection)"""
    Mr=[[0]*n for _ in range(n)]
    for j,b in enumerate(B):
        img=image_unsigned(b,*transf)
        if img is None: continue
        for k,co in img.items():
            if k in Bi: Mr[Bi[k]][j]=co%P
    return Mr
def proj_vec(combo):
    """project a dict {integral:coeff} onto numerator basis -> length-n vector"""
    v=[0]*n
    for k,co in combo.items():
        if k in Bi: v[Bi[k]]=(v[Bi[k]]+co)%P
    return v
def apply_transf_to_combo(combo,transf):
    if combo is None: return {}
    out={}
    for I,c in combo.items():
        img=image_unsigned(I,*transf)
        if img is None: continue
        for k,co in img.items(): out[k]=(out.get(k,0)+c*co)%P
    return out
def matmul(A,Bm):
    return [[sum(A[i][k]*Bm[k][j] for k in range(n))%P for j in range(n)] for i in range(n)]

# collect automorphism transforms of the sector
autos=[]
for t in _transforms(corner):
    imgZ=image_unsigned(corner,*t)
    if imgZ and len(imgZ)==1 and sec(next(iter(imgZ)))==csec:
        autos.append(t)
print(f"sector {S}: {len(autos)} automorphism transforms")

mats=[proj_action(t) for t in autos]
ident=[[1 if i==j else 0 for j in range(n)] for i in range(n)]

# (1) HOMOMORPHISM CHECK: M(s)M(t) == proj[ s(t(B)) ] for every basis vector, all pairs
hom_ok=True; checked=0
for a,ta in enumerate(autos):
    for b,tb in enumerate(autos):
        MM=matmul(mats[a],mats[b])                      # M(a)*M(b)
        for j,bv in enumerate(B):
            composed=apply_transf_to_combo(image_unsigned(bv,*tb), ta)   # a( b(B_j) )
            if proj_vec(composed)!=[MM[i][j] for i in range(n)]:
                hom_ok=False
        checked+=1
print(f"homomorphism M(s)M(t)==proj[s(t(.))] holds for all {checked} pairs: {hom_ok}")

# (2) exhibit an explicit non-commuting pair
found=None
for a in range(len(mats)):
    for b in range(a+1,len(mats)):
        if matmul(mats[a],mats[b])!=matmul(mats[b],mats[a]):
            found=(a,b); break
    if found: break
print(f"explicit non-commuting pair among automorphisms: {found}")
if found:
    a,b=found
    print(f"  transform A = {autos[a]}")
    print(f"  transform B = {autos[b]}")
    # show the 2x2 (or larger) block where they fail to commute: pick a basis vec
    for j in range(n):
        AB=[matmul(mats[a],mats[b])[i][j] for i in range(n)]
        BA=[matmul(mats[b],mats[a])[i][j] for i in range(n)]
        if AB!=BA:
            print(f"  on numerator slot {slots[j]}:  A*B col={AB}")
            print(f"                                B*A col={BA}")
            break

# (3) group order + abelian?  (multiply out closure)
def key(M): return tuple(tuple(r) for r in M)
gens=[m for m in mats if m!=ident]
G={key(ident)}; frontier=[ident]
while frontier:
    x=frontier.pop()
    for g in gens:
        y=matmul(x,g)
        if key(y) not in G: G.add(key(y)); frontier.append(y)
    if len(G)>2000: break
abelian=all(matmul(g1,g2)==matmul(g2,g1) for g1 in gens for g2 in gens)
print(f"generated matrix group order = {len(G)}   abelian = {abelian}")
print("VERDICT:", "genuine >=2D irrep -> irreducible residual (no term order triangularizes)"
      if not abelian else "abelian -> triangularizable")
