#!/usr/bin/env python
"""CORRECT replay: sector symmetric iff len(within-sector autos)>1 (autos FIX the corner
but act on numerators). Classify dispatched/passenger integrals free/survivor/asymmetric."""
import sys, os, pickle
from collections import Counter
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def secbits(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
def sector(i): return tuple(k for k in range(ND) if i[k]>0)
def corner_of(sec): return tuple(1 if k in sec else 0 for k in range(11))
def tw(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0), tuple(-abs(x) for x in i))  # LARGER = higher, matches ibp_env.weight
def inv(a): return pow(a,P-2,P)
_ac={}
def sym_autos(corner):
    if corner in _ac: return _ac[corner]
    cs=secbits(corner)
    a=[t for t in _transforms(corner)
       if (lambda im: im and len(im)==1 and secbits(next(iter(im)))==cs)(image_unsigned(corner,*t))]
    _ac[corner]=a; return a
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
def classify(I):
    autos=sym_autos(corner_of(sector(I)))
    if len(autos)<=1: return "asymmetric"
    rels=[]; ints={I}
    for t in autos:
        img=image_unsigned(I,*t)
        if img is None or img=={I:1}: continue
        rel={I:1}
        for k,c in img.items(): rel[k]=(rel.get(k,0)-c)%P; ints.add(k)
        rel={k:c for k,c in rel.items() if c%P}
        if rel: rels.append(rel)
    if not rels: return "sym_survivor"
    allints=sorted(ints,key=tw, reverse=True)
    red=rref([[r.get(ii,0)%P for ii in allints] for r in rels],len(allints))
    for row in red:
        pc=next((j for j in range(len(allints)) if row[j]%P),None)
        if pc is None or allints[pc]!=I: continue
        tail=[allints[j] for j in range(pc+1,len(allints)) if row[j]%P]
        return "sym_reducible_FREE" if all(tw(t2) < tw(I) for t2 in tail) else "sym_survivor"
    return "sym_survivor"

print("SANITY (want >1):", {s:len(sym_autos(corner_of(s))) for s in [(1,2,4,7),(0,3,4,5),(0,1,2,3)]})
runs=["probe_84_v7_successonly","probe_74_v7_successonly","probe_monster_v7_totalweight","probe_memhog_v7_totalweight"]
subs=set(); expr=set()
for r in runs:
    with open(f"results/{r}/result.pkl","rb") as f: d=pickle.load(f)
    for k in d.get('full_subs_replay',{}): subs.add(tuple(k))
    for k in d.get('full_expr_replay',{}): expr.add(tuple(k))
def report(name,pop):
    c=Counter(classify(I) for I in pop); tot=len(pop)
    print(f"\n=== {name}: {tot} ===")
    for k in ("sym_reducible_FREE","sym_survivor","asymmetric"):
        print(f"  {c[k]:4d} ({100*c[k]/max(tot,1):4.1f}%)  {k}")
report("reduced (top-worker targets)", subs)
report("passengers", expr-subs)
report("ALL touched", subs|expr)
