#!/usr/bin/env python
"""Aggregate the BULK workers, not the few outliers. For each target/mode report:
total CPU, CPU in the top-10 hardest, and the BULK CPU (total minus top-10) plus its
worker count and mean. The bulk metric shows whether the many ordinary workers win
even when a few hard survivors dominate the raw total."""
import pickle, glob, statistics as st
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/ab_symmetry"
def times(tag,mode):
    T=[]
    for p in glob.glob(f"{BASE}/{tag}/{mode}/work/results/*.pkl"):
        try:
            w=pickle.load(open(p,"rb"))
            if isinstance(w,dict) and 'time' in w: T.append(w['time'])
        except Exception: pass
    return sorted(T,reverse=True)
print(f"{'target':13s} {'mode':8s} {'nW':>5} {'totalCPU':>9} {'top10':>7} {'BULK(-top10)':>12} {'bulkN':>6} {'bulkMean':>9} {'p90-sumCPU':>11}")
print("-"*90)
for tag in ["gate_small","m2_4prop_dots","m1_6prop","m3_5prop_deg3"]:
    row={}
    for mode in ["baseline","design1"]:
        T=times(tag,mode)
        if not T: continue
        tot=sum(T); top10=sum(T[:10]); bulk=sum(T[10:]); bn=len(T)-10
        # p90: sum of the cheapest 90% of workers (drop the hardest 10%)
        k=max(1,int(len(T)*0.9)); p90=sum(sorted(T)[:k])
        bmean = (bulk/bn) if bn>0 else 0
        row[mode]=(len(T),tot,top10,bulk,bn,bmean,p90)
        print(f"{tag:13s} {mode:8s} {len(T):5d} {tot:9.0f} {top10:7.0f} {bulk:12.0f} {bn:6d} {bmean:9.2f} {p90:11.0f}")
    if 'baseline' in row and 'design1' in row:
        b,d=row['baseline'],row['design1']
        if b[3]>0: print(f"{'':22s}   -> BULK CPU design1/baseline = {100*d[3]/b[3]:.0f}%   |  p90 CPU = {100*d[6]/b[6]:.0f}%   |  count = {100*d[0]/b[0]:.0f}%")
    print()
