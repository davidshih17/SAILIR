#!/usr/bin/env python
"""Where do the HARD dots-only cases live in (L, d)?

Scans all campaign lists (longrunner v1/v2, memhog v1) + the g885_symfirst
pending stragglers, keeps pure-dot integrals (no numerator powers, i.e. no
negative entries; at least one entry >= 2), prints the (L, d) histogram and
each integral with its recorded runtime where available.
"""
import glob
import re

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
FILES = [
    ROOT + "/results/gr_reduce/g1023/longrunner_campaign_v1.txt",
    ROOT + "/results/gr_reduce/g1023/longrunner_campaign_v2.txt",
    ROOT + "/results/gr_reduce/g1023/memhog_campaign_v1.txt",
]

seen = {}
for fp in FILES:
    try:
        lines = open(fp).read().splitlines()
    except OSError:
        continue
    for ln in lines:
        if ln.startswith("#"):
            continue
        m = re.search(r"(-?\d+(?:,-?\d+){10,})", ln)
        if not m:
            continue
        t = tuple(int(x) for x in m.group(1).split(","))
        info = ln.replace(m.group(1), "").strip()
        seen.setdefault(t, []).append((fp.split("/")[-1], info))

hist = {}
dots = []
for t, srcs in seen.items():
    if any(x < 0 for x in t):
        continue
    L = sum(1 for x in t if x > 0)
    r = sum(x for x in t if x > 0)
    d = r - L
    if d == 0:
        continue                       # corner, not dotted
    dots.append((L, d, t, srcs))
    hist[(L, d)] = hist.get((L, d), 0) + 1

print(f"campaign integrals scanned: {len(seen)} unique; "
      f"pure-dot (d>=1): {len(dots)}")
print("\n(L, d) histogram of HARD dots-only cases:")
for (L, d) in sorted(hist):
    print(f"  L={L} d={d}: {hist[(L, d)]}")
print("\ndetails:")
for L, d, t, srcs in sorted(dots):
    tags = "; ".join(f"{f}: {i[:40]}" for f, i in srcs[:2])
    print(f"  L={L} d={d}  {','.join(map(str, t))}   [{tags}]")
