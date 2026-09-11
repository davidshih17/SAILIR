#!/usr/bin/env python
"""Show that the per-orbit linear solve yields INDIVIDUAL-integral rewrite rules
(integral -> strictly-lower individual integrals) + individual-integral masters --
never a symmetric combination as a master. Demonstrated on {0,3,4,5} and a survivor
sector, at degree-1 numerator level."""
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
def rref(rows,cols):
    rows=[r[:] for r in rows if any(x%P for x in r)]; piv=[]; r=0
    for c in range(cols):
        p=next((i for i in range(r,len(rows)) if rows[i][c]%P),None)
        if p is None: continue
        rows[r],rows[p]=rows[p],rows[r]; iv=inv(rows[r][c]); rows[r]=[(x*iv)%P for x in rows[r]]
        for i in range(len(rows)):
            if i!=r and rows[i][c]%P:
                f=rows[i][c]; rows[i]=[(rows[i][j]-f*rows[r][j])%P for j in range(cols)]
        piv.append(c); r+=1
        if r==len(rows): break
    return rows[:r],piv

def demo(combo):
    corner=tuple(1 if k in combo else 0 for k in range(11)); csec=sec(corner)
    slots=[j for j in range(ND) if j not in combo]+[8,9,10]
    B=[tuple(-1 if k==j else corner[k] for k in range(11)) for j in slots]
    Bi=set(B)
    autos=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and sec(next(iter(im)))==csec)(image_unsigned(corner,*t))]
    lvl=tw(B[0])[:2]
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
                if tw(k)[:2]==lvl and k not in seen and k in Bi: work.append(k)
    allints=sorted({k for r in rels for k in r}, key=tw, reverse=True)  # highest first
    mat=[[r.get(ii,0)%P for ii in allints] for r in rels]
    red,piv=rref(mat,len(allints))
    reduced=set(); rules=[]
    for row,pc in zip(red,piv):
        pivint=allints[pc]
        if tw(pivint)[:2]!=lvl or pivint not in Bi: continue
        tail=[(allints[j], (-row[j])%P) for j in range(pc+1,len(allints)) if row[j]%P]
        if all(tw(t2) < tw(pivint) for t2,_ in tail):
            reduced.add(pivint); rules.append((pivint,tail))
    masters=[b for b in B if b not in reduced]
    print(f"=== sector {combo} (degree-1 numerators, {len(B)} of them) ===")
    print(f"  INDIVIDUAL-integral masters kept: {len(masters)}")
    for m in masters: print(f"     master: {m}")
    print(f"  INDIVIDUAL-integral rewrite rules (num -> strictly-lower individual integrals): {len(rules)}")
    for pivint,tail in rules[:3]:
        rhs=" + ".join(f"{c}*{t2}" for t2,c in tail[:3])
        print(f"     {pivint}  ->  {rhs}{' + ...' if len(tail)>3 else ''}")
    print()

demo((0,3,4,5))
demo((0,1,2,3))
