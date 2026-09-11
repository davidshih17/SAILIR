#!/usr/bin/env python
"""Full distribution of per-worker time, baseline vs design1: percentiles + a bucketed
histogram (worker count and CPU-share per time bucket)."""
import pickle, glob, numpy as np
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/ab_symmetry"
def times(tag,mode):
    T=[]
    for p in glob.glob(f"{BASE}/{tag}/{mode}/work/results/*.pkl"):
        try:
            w=pickle.load(open(p,"rb"))
            if isinstance(w,dict) and 'time' in w: T.append(w['time'])
        except Exception: pass
    return np.array(sorted(T))
EDGES=[0,1,3,10,30,100,300,10**9]; LBL=["<1s","1-3s","3-10s","10-30s","30-100s","100-300s",">300s"]
for tag in ["m2_4prop_dots","m1_6prop","m3_5prop_deg3"]:
    print(f"================= {tag} =================")
    for mode in ["baseline","design1"]:
        T=times(tag,mode)
        if len(T)==0: continue
        pcts=np.percentile(T,[50,75,90,95,99])
        print(f"  [{mode}] nW={len(T)} totalCPU={T.sum():.0f}s")
        print(f"     percentiles(s): p50={pcts[0]:.1f} p75={pcts[1]:.1f} p90={pcts[2]:.1f} p95={pcts[3]:.1f} p99={pcts[4]:.1f} max={T.max():.0f}")
        print(f"     {'bucket':>10}: {'nWork':>6} ({'%cnt':>4})  {'CPU(s)':>8} ({'%cpu':>4})")
        for i,lab in enumerate(LBL):
            m=(T>=EDGES[i])&(T<EDGES[i+1]); n=int(m.sum()); c=float(T[m].sum())
            if n: print(f"     {lab:>10}: {n:6d} ({100*n/len(T):4.0f}) {c:8.0f} ({100*c/T.sum():4.0f})")
    print()
