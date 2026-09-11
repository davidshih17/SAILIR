#!/usr/bin/env python
"""For each running arm: list PENDING worker integrals (log exists, no result
pkl), their runtime, and proximity to masters/corners of the 142 basis:
  MASTER            exactly a basis element
  SYM-of-MASTER     symmetry image of a basis element (canonical-orbit map)
  CORNER            corner of its sector (all indices 1, no dots/numerators)
  corner+kd,ms      k extra dots, m numerator powers away from the corner
"""
import os, sys, glob, time, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
import topo_config as _tc

masters = set()
for ln in open(os.path.join(ROOT, "results/gr_cut_masters_142.txt")):
    masters.add(tuple(int(x) for x in ln.split(",")))


def sec(t):
    return sum(1 << i for i in range(_tc.N_DEN) if t[i] > 0)


now = time.time()
for tag in ("g885", "g127", "g893"):
    d = os.path.join(ROOT, f"results/gr_reduce/{tag}/work")
    pend = []
    for lg in glob.glob(d + "/logs/*.out"):
        base = os.path.basename(lg)[:-4]
        if not os.path.exists(d + f"/results/{base}.pkl"):
            t = tuple(int(x) for x in base.split("_")[2:])
            pend.append((now - os.path.getmtime(lg.replace(".out", ".log")),
                         (now - os.path.getctime(lg)) / 60, t))
    print(f"\n=== {tag}: {len(pend)} pending")
    for _, age_min, t in sorted(pend, key=lambda x: -x[1]):
        dots = sum(x - 1 for x in t[:_tc.N_DEN] if x > 1)
        s = sum(-x for x in t if x < 0)
        S = sec(t)
        if t in masters:
            label = "** MASTER **"
        elif dots == 0 and s == 0:
            label = "CORNER" + (" (=master's sector!)" if any(
                sec(m) == S for m in masters) else "")
        else:
            label = f"corner+{dots}d,{s}n"
        n_sec_masters = sum(1 for m in masters if sec(m) == S)
        print(f"  {age_min:7.1f} min  {list(t)}  sec={S} "
              f"(sector has {n_sec_masters} masters)  {label}")
