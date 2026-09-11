#!/usr/bin/env python
"""ENHANCED CACHE HIT RATE: if we mined the upward excursions out of each
one-level worker, how many of them does the campaign ACTUALLY still need?

"Not cached and not a master" overstates it -- some upward excursions are
integrals the reduction never needs again. The real measure is the intersection
with the LIVE frontier: the non-masters in the current expression.

Rebuilds the expression exactly as the orchestrator's resume does (load every
result into a cache, fold from the start integral), then samples workers and
asks what fraction of their above-T eliminations land in that frontier.
"""
import os, pickle, random, sys, time
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
sys.path.insert(0, '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/reduction')
from sailir.topology import Topology
from sailir import ibp_env as ie
import topo_config as tc
ie.init_from_topology(Topology.from_dir(tc.TOPO_DIR)); ie.set_prime(101)
ie.set_paper_masters_only(False)
from canonical_masters import apply_canonical_masters; apply_canonical_masters()
from sailir.ibp_env import is_master
from total_order import tkey
from hierarchical_reduction import apply_substitutions, get_non_masters, coarse_weight

D = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/gr_reduce/g1023_greedy_certified/work/results'
START = (1,1,1,1,1,1,1,1,2,2,0,0,0,0,0)

t0=time.time(); cache={}; names=[n for n in os.listdir(D) if n.endswith('.pkl')]
print(f"  result files: {len(names):,}", flush=True)
for i,n in enumerate(names):
    try:
        with open(os.path.join(D,n),'rb') as f: d=pickle.load(f)
    except Exception: continue
    oi=d.get('original_integral')
    if oi is None: continue
    oi=tuple(oi)
    cache[oi]= d.get('final_expr', {oi:1}) if d.get('success') else {oi:1}
    if (i+1)%100000==0: print(f"    ... {i+1:,} loaded, {time.time()-t0:.0f}s", flush=True)
print(f"  cache {len(cache):,} in {time.time()-t0:.0f}s", flush=True)

t1=time.time()
expr = apply_substitutions({START:1}, cache, 101, progress=200_000)
print(f"  fold {time.time()-t1:.0f}s -> |expr|={len(expr):,}", flush=True)
frontier = get_non_masters(expr)
print(f"  LIVE FRONTIER (non-masters in expr): {len(frontier):,}", flush=True)

random.Random(0).shuffle(names)
n_w=0; steps=0; above=set()
for n in names[:2000]:
    try:
        with open(os.path.join(D,n),'rb') as f: d=pickle.load(f)
    except Exception: continue
    if not d.get('success'): continue
    p=d.get('path') or []
    if len(p)<2: continue
    T=tuple(d['original_integral']); kT=tkey(T)
    n_w+=1; steps+=len(p)
    for e in p:
        try: J=tuple(e[0])
        except Exception: continue
        if tkey(J) < kT: above.add(J)       # smaller tkey = higher
print(f"\n  workers sampled: {n_w:,}  mean steps {steps/max(n_w,1):.0f}", flush=True)
print(f"  distinct upward excursions: {len(above):,}", flush=True)
inc = above & frontier
cached_already = sum(1 for i in above if i in cache)
print(f"    already cached                 : {cached_already:,} ({100*cached_already/len(above):.1f}%)")
print(f"    IN THE LIVE FRONTIER (the prize): {len(inc):,} ({100*len(inc)/len(above):.1f}%)")
print(f"    neither (never needed)          : {len(above)-cached_already-len(inc):,}")
print(f"  => per worker: {len(inc)/max(n_w,1):.1f} reductions the campaign STILL NEEDS")
print(f"  => frontier is {len(frontier):,}; mining {n_w:,} workers would cover "
      f"{100*len(inc)/max(len(frontier),1):.2f}% of it", flush=True)
if inc:
    from collections import Counter
    c=Counter(coarse_weight(i)[:2] for i in inc)
    print("  coverage by (L,r):", ' '.join(f"L{L}r{r}:{n}" for (L,r),n in sorted(c.items(), reverse=True)[:10]), flush=True)
