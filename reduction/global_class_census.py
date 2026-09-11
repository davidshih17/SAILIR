#!/usr/bin/env python
"""GLOBAL census of training-decision state classes across ALL sectors of
gravity3L_canon10x: counts by (pure-dot / pure-numerator / mixed / corner)
and by (dots, numerator-depth) class, over every decision in the raw jsonl."""
import glob
import json
from collections import Counter

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
N_DEN = 10

cls = Counter()
depth = Counter()
n = 0
for f in sorted(glob.glob(ROOT + "/data/gravity3L_canon10x_raw_jsonl/*.jsonl")):
    for line in open(f):
        try:
            t = json.loads(line)["target"]
        except (json.JSONDecodeError, KeyError):
            continue
        n += 1
        dots = sum(x - 1 for x in t[:N_DEN] if x > 1)
        s = sum(-x for x in t if x < 0)
        if dots and s:
            cls["MIXED"] += 1
        elif dots:
            cls["PURE-DOT"] += 1
            depth[("d", dots)] += 1
        elif s:
            cls["PURE-NUMERATOR"] += 1
            depth[("s", s)] += 1
        else:
            cls["CORNER"] += 1

print(f"total decisions: {n}")
for k, v in cls.most_common():
    print(f"  {k:15s} {v:9d}  ({100.0*v/n:.3f}%)")
print("\npure-dot by depth:    " +
      "  ".join(f"d{k[1]}:{v}" for k, v in sorted(depth.items())
                if k[0] == "d"))
print("pure-numerator by depth: " +
      "  ".join(f"s{k[1]}:{v}" for k, v in sorted(depth.items())
                if k[0] == "s"))
