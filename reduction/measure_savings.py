#!/usr/bin/env python
"""Accurate savings: classify EVERY integral the orchestrator actually reduced (the
103k reduction_cache keys) into FREE (symmetry rewrite rule -> no worker) vs WORKER
(survivor or asymmetric). The free fraction = work symmetry saves vs the original orch."""
import sys, os, pickle, time
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
def pc(i): return sum(1 for k in range(ND) if i[k]>0)
def w12(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0))
def tw_desc(i): w=w12(i); return (-w[0],-w[1],tuple(abs(x) for x in i))
def corner_of(sec): return tuple(1 if k in sec else 0 for k in range(11))
def inv(a): return pow(a,P-2,P)
_ac={}
def sym_autos(corner):
    if corner in _ac: return _ac[corner]
    cs=secbits(corner); a=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and secbits(next(iter(im)))==cs)(image_unsigned(corner,*t))]
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
    if len(autos)<=1: return "asym"
    rels=[]; ints={I}
    for t in autos:
        img=image_unsigned(I,*t)
        if img is None or img=={I:1}: continue
        rel={I:1}
        for k,c in img.items(): rel[k]=(rel.get(k,0)-c)%P; ints.add(k)
        rel={k:c for k,c in rel.items() if c%P}
        if rel: rels.append(rel)
    if not rels: return "survivor"
    allints=sorted(ints,key=tw_desc)
    red=rref([[r.get(ii,0)%P for ii in allints] for r in rels],len(allints))
    for row in red:
        pcol=next((j for j in range(len(allints)) if row[j]%P),None)
        if pcol is None or allints[pcol]!=I: continue
        tail=[allints[j] for j in range(pcol+1,len(allints)) if row[j]%P]
        return "free" if all(tw_desc(t2)>tw_desc(I) for t2 in tail) else "survivor"
    return "survivor"

cache=pickle.load(open("replay/reduction_cache.pkl","rb")); CACHE=cache.get('cache',cache)
keys=list(CACHE.keys())
print(f"integrals the orchestrator reduced (cache keys): {len(keys)}", flush=True)
cat=Counter(); bypc=Counter(); t0=time.perf_counter()
for n,I in enumerate(keys):
    c=classify(I); cat[c]+=1; bypc[(pc(I),c)]+=1
    if (n+1)%20000==0: print(f"  ...{n+1} classified ({time.perf_counter()-t0:.0f}s)", flush=True)
tot=len(keys)
print(f"\n=== SAVINGS over the full orchestrator work ({tot} reduced integrals) ===")
print(f"  FREE (symmetry rule, no worker): {cat['free']:6d}  ({100*cat['free']/tot:.1f}%)")
print(f"  survivor (symmetric, worker):    {cat['survivor']:6d}  ({100*cat['survivor']/tot:.1f}%)")
print(f"  asymmetric (worker):             {cat['asym']:6d}  ({100*cat['asym']/tot:.1f}%)")
print(f"\n  => symmetry would save {100*cat['free']/tot:.1f}% of the reduced-integral work")
print(f"     workers still needed: {cat['survivor']+cat['asym']} ({100*(cat['survivor']+cat['asym'])/tot:.1f}%)")
print("\nby propagator count:")
for p in sorted({k[0] for k in bypc}):
    f=bypc[(p,'free')]; s=bypc[(p,'survivor')]; a=bypc[(p,'asym')]; t=f+s+a
    print(f"  {p}-prop: {t:6d} reduced | FREE {f:6d} ({100*f/max(t,1):4.0f}%)  worker {s+a:6d}")
