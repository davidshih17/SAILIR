#!/usr/bin/env python
"""Replicate the WORKER exactly (route + composed) with instrumentation.

leak_probe.py showed bare _route is fast and flat, so if the worker is slow the
cost is in composed(). Note MAX_TERMS caps only the RAW rule; the COMPOSED rule
is uncapped, and composed() substitutes each of a rule's terms with that term's
own rule recursively -- so rule size can multiply with cascade depth while comp
caches every intermediate.
"""
import os, sys, time, resource
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from symmetry_route import canonical_monolithic_rule as _route
import topo_config as _tc
P = _tc.canonicalize_module().P
def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

_MAX_TERMS = int(os.environ.get('SAILIR_ROUTE_MAX_TERMS','200'))
_MAX_S = int(os.environ.get('SAILIR_ROUTE_MAX_S','5'))
raw={}; comp={}
def route(i):
    if i not in raw:
        if _MAX_S>=0 and -sum(x for x in i if x<0)>_MAX_S: raw[i]=None; return None
        r=_route(i)
        if r is not None and len(r)>_MAX_TERMS: r=None
        raw[i]=r
    return raw[i]

def composed(top):
    stack=[top]
    while stack:
        j=stack[-1]
        if j in comp: stack.pop(); continue
        r=route(j)
        deps=[k for k in r if k not in comp and route(k) is not None]
        if deps: stack.extend(deps); continue
        out={}
        for k,c in r.items():
            sub=comp[k] if route(k) is not None else None
            if sub is None: out[k]=(out.get(k,0)+c)%P
            else:
                for l,cl in sub.items():
                    v=(out.get(l,0)+c*cl)%P
                    if v: out[l]=v
                    else: out.pop(l,None)
        comp[j]={k:v for k,v in out.items() if v}
        stack.pop()
    return comp[top]

integrals=[]
for ln in open(sys.argv[1]):
    ln=ln.strip()
    if ln: integrals.append(tuple(int(x) for x in ln.split(",")))
print(f"batch {os.path.basename(sys.argv[1])}: {len(integrals)} integrals  P={P}",flush=True)
t0=time.time(); nr=0
for n,I in enumerate(integrals,1):
    ts=time.time()
    r=route(I)
    if r is None: sz=-1
    else:
        sz=len(composed(I)); nr+=1
    dt=time.time()-ts
    biggest=max((len(v) for v in comp.values()), default=0)
    tot=sum(len(v) for v in comp.values())
    print(f"  {n:3d}/{len(integrals)} s={-sum(x for x in I if x<0)} "
          f"composed_terms={sz:8d} {dt:8.2f}s rss={rss_mb():8.1f}MB "
          f"|raw|={len(raw):6d} |comp|={len(comp):6d} max_rule={biggest:8d} "
          f"sum_rule_terms={tot:9d} total={time.time()-t0:6.0f}s",flush=True)
print(f"DONE {time.time()-t0:.0f}s rss={rss_mb():.1f}MB routable={nr}",flush=True)
