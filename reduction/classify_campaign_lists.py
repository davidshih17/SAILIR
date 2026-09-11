#!/usr/bin/env python
"""Classify the campaign integrals (long-runner + memhog lists) against the
two training-desert categories identified from the coverage analysis:
  PURE-NUMERATOR: no dots (every den power <= 1), >=1 numerator power
  PURE-DOT:       >=1 dot, no numerator powers
  MIXED:          dots AND numerators (the well-covered training regime)
  CORNER:         neither (plain corner — shouldn't appear; would be a master)
Also reports sector distribution and overlap between the two lists.
"""
import os
import re
import sys
from collections import Counter

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
G = os.path.join(ROOT, "results/gr_reduce/g1023")
N_DEN = 10

def parse(path, int_col_last=True):
    out = []
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        m = re.search(r"([0-9-]+(?:,[0-9-]+){14})\s*$", ln)
        if m:
            out.append(tuple(int(x) for x in m.group(1).split(",")))
    return out

longr = parse(os.path.join(G, "longrunner_campaign_v1.txt"))
memh = parse(os.path.join(G, "memhog_campaign_v1.txt"))
both = set(longr) | set(memh)
overlap = set(longr) & set(memh)


def classify(t):
    dots = sum(x - 1 for x in t[:N_DEN] if x > 1)
    s = sum(-x for x in t if x < 0)
    if dots == 0 and s > 0:
        return "PURE-NUMERATOR"
    if dots > 0 and s == 0:
        return "PURE-DOT"
    if dots > 0 and s > 0:
        return "MIXED"
    return "CORNER"


def sec(t):
    return sum(1 << i for i in range(N_DEN) if t[i] > 0)


for name, lst in [("long-runner", set(longr)), ("memhog", set(memh)),
                  ("UNION", both)]:
    c = Counter(classify(t) for t in lst)
    n = len(lst)
    parts = "  ".join(f"{k}: {v} ({100*v//n}%)" for k, v in c.most_common())
    print(f"{name:12s} n={n:4d}  {parts}")
print(f"\noverlap between lists: {len(overlap)} integrals")

print("\ntop sectors (union):")
for s, n in Counter(sec(t) for t in both).most_common(10):
    cls = Counter(classify(t) for t in both if sec(t) == s)
    print(f"  sector {s:4d}: {n:3d}   " +
          "  ".join(f"{k}:{v}" for k, v in cls.most_common()))

# MIXED entries: how deep? (the coverage analysis found mixed classes well
# covered UNLESS very deep)
mixed = [t for t in both if classify(t) == "MIXED"]
if mixed:
    from collections import Counter as C2
    depth = Counter()
    for t in mixed:
        dots = sum(x - 1 for x in t[:N_DEN] if x > 1)
        s = sum(-x for x in t if x < 0)
        depth[(dots, s)] += 1
    print("\nMIXED (dots,s) breakdown: " +
          "  ".join(f"d{k[0]}s{k[1]}:{v}" for k, v in sorted(depth.items())))
