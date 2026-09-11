#!/usr/bin/env python
"""Replay: over real v7 reductions, classify every DISPATCHED (reduced) integral as
  - asymmetric sector          -> needs a worker (symmetry can't help)
  - symmetric sector, survivor -> needs a worker
  - symmetric sector, reducible-> FREE (a symmetry rewrite rule handles it, no worker)
Counts the fraction of workers Design 1 would eliminate."""
import sys, os, pickle, itertools
from collections import Counter
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def pc(i): return sum(1 for k in range(ND) if i[k]>0)
def tw(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0), tuple(-abs(x) for x in i))  # LARGER = higher, matches ibp_env.weight
def inv(a): return pow(a,P-2,P)
def rref(rows,w):
    rows=[r[:] for r in rows if any(x%P for x in r)]; piv=[]; r=0
    for c in range(w):
        p=next((i for i in range(r,len(rows)) if rows[i][c]%P),None)
        if p is None: continue
        rows[r],rows[p]=rows[p],rows[r]; ivv=inv(rows[r][c]); rows[r]=[(x*ivv)%P for x in rows[r]]
        for i in range(len(rows)):
            if i!=r and rows[i][c]%P:
                f=rows[i][c]; rows[i]=[(rows[i][j]-f*rows[r][j])%P for j in range(w)]
        piv.append((c,r)); r+=1
        if r==len(rows): break
    return rows[:r]

def classify(I):
    ts=_transforms(I)
    rels=[]; ints={I}
    for t in ts:
        img=image_unsigned(I,*t)
        if img is None or img=={I:1}: continue
        rel={I:1}
        for k,c in img.items(): rel[k]=(rel.get(k,0)-c)%P; ints.add(k)
        rel={k:c for k,c in rel.items() if c%P}
        if rel: rels.append(rel)
    if not rels: return "asymmetric"
    allints=sorted(ints, key=tw, reverse=True)
    mat=[[r.get(ii,0)%P for ii in allints] for r in rels]
    red=rref(mat,len(allints))
    # reducible iff some row pivots on I with a strictly-lower-total-weight tail
    for row in red:
        pcol=next((j for j in range(len(allints)) if row[j]%P),None)
        if pcol is None or allints[pcol]!=I: continue
        tail=[allints[j] for j in range(pcol+1,len(allints)) if row[j]%P]
        if all(tw(t2) < tw(I) for t2 in tail):
            return "sym_reducible_FREE"
        return "sym_survivor"
    return "sym_survivor"

runs=["probe_84_v7_successonly","probe_74_v7_successonly","probe_monster_v7_totalweight","probe_memhog_v7_totalweight"]
pop=set()
for r in runs:
    with open(f"results/{r}/result.pkl","rb") as f: d=pickle.load(f)
    for k in d.get('full_subs_replay',{}): pop.add(tuple(k))
print(f"dispatched (reduced) integrals across {len(runs)} real v7 reductions: {len(pop)} distinct\n")
cat=Counter(); bypc=Counter()
for I in pop:
    c=classify(I); cat[c]+=1; bypc[(pc(I),c)]+=1
tot=len(pop)
print("=== classification of dispatched integrals ===")
for c in ("sym_reducible_FREE","sym_survivor","asymmetric"):
    print(f"  {cat[c]:4d} ({100*cat[c]/tot:4.1f}%)  {c}")
print(f"\n  => Design 1 would route {cat['sym_reducible_FREE']} of {tot} "
      f"({100*cat['sym_reducible_FREE']/tot:.1f}%) for FREE (no worker)")
print("\n=== by propagator count (where the free routing lives) ===")
for p in sorted({k[0] for k in bypc}):
    free=bypc[(p,'sym_reducible_FREE')]; surv=bypc[(p,'sym_survivor')]; asym=bypc[(p,'asymmetric')]
    t=free+surv+asym
    print(f"  {p}-prop: {t:4d} dispatched | FREE {free:3d}  survivor {surv:3d}  asymmetric {asym:3d}")
