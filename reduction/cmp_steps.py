#!/usr/bin/env python
"""Is design1 harder (more STEPS) or slower per step (overhead)? Compare per-worker
step count and time/step, baseline vs design1. Also check the median (typical) worker
vs the mean (outlier-driven)."""
import pickle, glob, statistics as st
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/ab_symmetry"
def load(tag,mode):
    W=[]
    for p in glob.glob(f"{BASE}/{tag}/{mode}/work/results/*.pkl"):
        try:
            w=pickle.load(open(p,"rb"))
            if isinstance(w,dict) and 'time' in w and 'steps' in w:
                W.append((w['steps'], w['time']))
        except Exception: pass
    return W
def line(tag,mode,W):
    steps=[s for s,_ in W]; times=[t for _,t in W]
    tps=[t/s for s,t in W if s>0]           # time per step
    print(f"{tag:12s} {mode:8s} nW={len(W):5d} | steps med={st.median(steps):4.0f} mean={st.mean(steps):6.1f} max={max(steps):5d}"
          f" | time/step med={st.median(tps):5.2f}s mean={st.mean(tps):6.2f}s"
          f" | steps==0: {sum(1 for s in steps if s==0)}")
for tag in ["m2_4prop_dots","m1_6prop","m3_5prop_deg3"]:
    for mode in ["baseline","design1"]:
        W=load(tag,mode)
        if W: line(tag,mode,W)
    print()
# where does the CPU actually go? share of total time in the top-10 hardest workers
print("=== CPU concentration: share of total worker-time in the 10 longest workers ===")
for tag in ["m2_4prop_dots"]:
    for mode in ["baseline","design1"]:
        W=load(tag,mode); times=sorted((t for _,t in W),reverse=True)
        top10=sum(times[:10]); tot=sum(times)
        print(f"  {tag} {mode}: top-10 workers = {top10:.0f}s / {tot:.0f}s total = {100*top10/tot:.0f}%")
