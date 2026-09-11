#!/usr/bin/env python
"""Print the LITERAL data: for num@1 in sector {0,3,4,5}, show each symmetry image
term with its (w1,w2) weight, and the row-reduced result — classifying tail terms by
(w1,w2) ONLY (not |abs|). Question: does symmetry actually push num@1 to strictly-lower
(w1,w2), or does it only relate it to OTHER same-(w1,w2) integrals?"""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def sec(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
def w12(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0))   # (w1,w2) ONLY
def inv(a): return pow(a,P-2,P)
combo=(0,3,4,5); corner=tuple(1 if k in combo else 0 for k in range(11))
csec=sec(corner)
u=tuple(-1 if k==1 else corner[k] for k in range(11))   # num@1
print(f"seed num@1 = {u}   (w1,w2) = {w12(u)}   sector {combo}\n")

autos=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and sec(next(iter(im)))==csec)(image_unsigned(corner,*t))]
print(f"{len(autos)} sector automorphisms. Images of num@1 (term : (w1,w2)):")
for idx,t in enumerate(autos[:6]):
    img=image_unsigned(u,*t)
    if img is None: continue
    same=[k for k in img if w12(k)==w12(u)]
    lower=[k for k in img if w12(k)!=w12(u) and w12(k)<w12(u)]
    print(f"  sym#{idx}: {len(img)} terms | same-(w1,w2): {len(same)}  strictly-lower-(w1,w2): {len(lower)}")
    for k,c in sorted(img.items()):
        tag = "SAME" if w12(k)==w12(u) else ("lower" if w12(k)<w12(u) else "HIGHER?!")
        print(f"       {k}  coeff={c%P:>4}  w12={w12(k)}  {tag}")
    print()

# Now the full reduction: collect all relations among the orbit, row-reduce, and report
# whether num@1 ends up expressed in STRICTLY-LOWER (w1,w2) or in same-(w1,w2) integrals.
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
rels=[]; seen=set(); work=[u]
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
            if w12(k)==w12(u) and k not in seen: work.append(k)
# order columns: same-(w1,w2) first (candidates to eliminate), then strictly lower
allints=sorted({k for r in rels for k in r}, key=lambda k:(w12(k)==w12(u),), reverse=True)
allints=sorted(allints, key=lambda k: w12(k), reverse=True)
col={ii:j for j,ii in enumerate(allints)}
mat=[[r.get(ii,0)%P for ii in allints] for r in rels]
red=rref(mat,len(allints))
sameweight=[k for k in allints if w12(k)==w12(u)]
print(f"total same-(w1,w2) integrals in the orbit system: {len(sameweight)}")
# for num@1 specifically: find the row that pivots on num@1
elim_same=0; elim_to_strictly_lower=0
for row in red:
    pc=next((j for j in range(len(allints)) if row[j]%P),None)
    if pc is None: continue
    piv=allints[pc]
    if w12(piv)!=w12(u): continue
    tail=[allints[j] for j in range(pc+1,len(allints)) if row[j]%P]
    tail_same=[t2 for t2 in tail if w12(t2)==w12(u)]
    tail_lower=[t2 for t2 in tail if w12(t2)<w12(u)]
    elim_same+=1
    if not tail_same: elim_to_strictly_lower+=1
    if piv==u:
        print(f"\nnum@1 pivot row: tail has {len(tail_same)} SAME-(w1,w2) terms, {len(tail_lower)} strictly-lower")
        print(f"  -> num@1 = {'STRICTLY-LOWER only (genuine weight reduction)' if not tail_same else 'still involves OTHER same-(w1,w2) integrals (NOT a weight reduction)'}")
print(f"\nsummary: {elim_same} same-weight integrals pivoted; {elim_to_strictly_lower} of them go to STRICTLY-lower (w1,w2) with NO same-weight tail")
