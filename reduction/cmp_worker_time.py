#!/usr/bin/env python
"""Compare baseline vs design1 on (a) total worker CPU time and (b) per-worker time
distribution -- does symmetry leave HARDER integrals for the workers?"""
import pickle, glob, os, statistics as st
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/ab_symmetry"
def stats(tag,mode):
    dd=f"{BASE}/{tag}/{mode}"
    times=[]
    for p in glob.glob(f"{dd}/work/results/*.pkl"):
        try:
            w=pickle.load(open(p,"rb"))
            if isinstance(w,dict) and 'time' in w: times.append(w['time'])
        except Exception: pass
    res=None
    rp=f"{dd}/reduction.pkl"
    if os.path.exists(rp):
        r=pickle.load(open(rp,"rb")); res=(r.get('total_worker_time'),r.get('elapsed_time'),r.get('total_jobs'))
    return times,res
print(f"{'target':14s} {'mode':8s} {'nW':>5} {'sumCPU(s)':>10} {'med(s)':>7} {'mean(s)':>7} {'max(s)':>8} {'elapsed(s)':>10}")
print("-"*78)
for tag in ["gate_small","m2_4prop_dots","m1_6prop","m3_5prop_deg3"]:
    for mode in ["baseline","design1"]:
        times,res=stats(tag,mode)
        if not times: print(f"{tag:14s} {mode:8s}  (no data)"); continue
        sumc=sum(times); med=st.median(times); mean=st.mean(times); mx=max(times)
        elapsed = res[1] if res else None
        done = "" if res else "  [running-partial]"
        print(f"{tag:14s} {mode:8s} {len(times):5d} {sumc:10.0f} {med:7.1f} {mean:7.1f} {mx:8.0f} "
              f"{('%.0f'%elapsed) if elapsed else '   --':>10}{done}")
    print()
