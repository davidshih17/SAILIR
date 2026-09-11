#!/usr/bin/env python
"""How much of the 24% free-routing comes from full-M (momentum-map) records vs the
placeholders? Recompute over the 103k cache using ONLY full-M autos, compare to all-autos."""
import sys, os, pickle, time
from collections import Counter
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8; N=11
def secbits(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
def sector(i): return tuple(k for k in range(ND) if i[k]>0)
def w12(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0))
def tw_desc(i): w=w12(i); return (-w[0],-w[1],tuple(abs(x) for x in i))
def corner_of(sec): return tuple(1 if k in sec else 0 for k in range(11))
def inv(a): return pow(a,P-2,P)
_ac={}
def autos_of(corner, fullM_only):
    key=(corner,fullM_only)
    if key in _ac: return _ac[key]
    cs=secbits(corner); a=[]
    for (M,c) in _transforms(corner):
        im=image_unsigned(corner,M,c)
        if not (im and len(im)==1 and secbits(next(iter(im)))==cs): continue
        if fullM_only and not all(i in M for i in range(N)): continue
        a.append((M,c))
    _ac[key]=a; return a
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
def is_free(I, fullM_only):
    autos=autos_of(corner_of(sector(I)), fullM_only)
    if len(autos)<=1: return False
    rels=[]; ints={I}
    for (M,c) in autos:
        img=image_unsigned(I,M,c)
        if img is None or img=={I:1}: continue
        rel={I:1}
        for k,co in img.items(): rel[k]=(rel.get(k,0)-co)%P; ints.add(k)
        rel={k:co for k,co in rel.items() if co%P}
        if rel: rels.append(rel)
    if not rels: return False
    allints=sorted(ints,key=tw_desc)
    red=rref([[r.get(ii,0)%P for ii in allints] for r in rels],len(allints))
    for row in red:
        pc=next((j for j in range(len(allints)) if row[j]%P),None)
        if pc is None or allints[pc]!=I: continue
        tail=[allints[j] for j in range(pc+1,len(allints)) if row[j]%P]
        return all(tw_desc(t2)>tw_desc(I) for t2 in tail)
    return False
cache=pickle.load(open("replay/reduction_cache.pkl","rb")); CACHE=cache.get('cache',cache)
keys=list(CACHE.keys()); tot=len(keys)
t0=time.perf_counter()
free_all=free_full=0; only_ph=[]
for n,I in enumerate(keys):
    fa=is_free(I,False); ff=is_free(I,True)
    free_all+= 1 if fa else 0; free_full+= 1 if ff else 0
    if fa and not ff and len(only_ph)<5: only_ph.append(I)
    if (n+1)%25000==0: print(f"  ...{n+1} ({time.perf_counter()-t0:.0f}s)", flush=True)
print(f"\nfree with ALL records (incl placeholders): {free_all} ({100*free_all/tot:.1f}%)")
print(f"free with FULL-M records only:             {free_full} ({100*free_full/tot:.1f}%)")
print(f"placeholders' UNIQUE contribution: {free_all-free_full} ({100*(free_all-free_full)/tot:.2f}%)")
print(f"duplication: full-M covers {100*free_full/max(free_all,1):.1f}% of what all-records covers")
print(f"\nexamples the placeholders uniquely made 'free': {only_ph}")
