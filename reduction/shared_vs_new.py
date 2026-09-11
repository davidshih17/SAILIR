#!/usr/bin/env python
"""Split design1's workers into those on integrals the BASELINE also reduced (shared)
vs integrals only design1's symmetry path reached (new). Where is the cost? This
distinguishes 'harder because different path introduces new hard integrals' from
'harder because same hard cores, cheap bulk removed'."""
import pickle, glob, statistics as st
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/ab_symmetry"
def wset(tag,mode):
    d={}
    for p in glob.glob(f"{BASE}/{tag}/{mode}/work/results/*.pkl"):
        try:
            w=pickle.load(open(p,"rb"))
            if isinstance(w,dict) and 'time' in w: d[tuple(w['original_integral'])]=w['time']
        except Exception: pass
    return d
for tag in ["m2_4prop_dots","m3_5prop_deg3"]:
    base=wset(tag,"baseline"); des=wset(tag,"design1")
    bkeys=set(base)
    shared=[(I,t) for I,t in des.items() if I in bkeys]
    new=[(I,t) for I,t in des.items() if I not in bkeys]
    st_share=[t for _,t in shared]; st_new=[t for _,t in new]
    print(f"=== {tag} ===")
    print(f"  baseline workers: {len(base)}   design1 workers: {len(des)}")
    if st_share: print(f"  design1 workers SHARED w/ baseline: {len(st_share):4d} | CPU sum={sum(st_share):7.0f}s med={st.median(st_share):.2f}s max={max(st_share):.0f}s")
    if st_new:   print(f"  design1 workers NEW (symmetry path): {len(st_new):4d} | CPU sum={sum(st_new):7.0f}s med={st.median(st_new):.2f}s max={max(st_new):.0f}s")
    # for shared integrals, did they cost the SAME in baseline vs design1? (same code, same integral)
    if shared:
        diffs=[abs(des[I]-base[I]) for I,_ in shared if base[I]>0]
        print(f"  shared integrals: |design1_time - baseline_time| med={st.median(diffs):.2f}s (should be ~0: same integral,same worker)")
    print()
