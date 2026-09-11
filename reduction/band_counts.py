#!/usr/bin/env python
"""Worker COUNT and CPU in three bands: cheap (<3s), medium (3-100s), hard tail (>100s),
with the hard tail split into 100-300s and >300s. baseline vs design1."""
import pickle, glob, numpy as np
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/ab_symmetry"
def times(tag,mode):
    T=[]
    for p in glob.glob(f"{BASE}/{tag}/{mode}/work/results/*.pkl"):
        try:
            w=pickle.load(open(p,"rb"))
            if isinstance(w,dict) and 'time' in w: T.append(w['time'])
        except Exception: pass
    return np.array(T)
def band(T,lo,hi): m=(T>=lo)&(T<hi); return int(m.sum()), float(T[m].sum())
print(f"{'target/mode':22s} | {'cheap<3s':>10} | {'medium 3-100s':>16} | {'tail 100-300s':>16} | {'tail >300s':>16} | {'maxCPU':>7}")
print("-"*104)
for tag in ["m2_4prop_dots","m1_6prop","m3_5prop_deg3"]:
    for mode in ["baseline","design1"]:
        T=times(tag,mode)
        if len(T)==0: continue
        cn,cc=band(T,0,3); mn,mc=band(T,3,100); h1n,h1c=band(T,100,300); h2n,h2c=band(T,300,1e18)
        print(f"{tag[:12]+'/'+mode[:3]:22s} | {cn:5d} {'':4s} | {mn:5d}w {mc:8.0f}s | {h1n:5d}w {h1c:8.0f}s | {h2n:5d}w {h2c:8.0f}s | {T.max():7.0f}")
    print("-"*104)
