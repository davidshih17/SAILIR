#!/usr/bin/env python
"""Do the truth-path states the model mis-ranks lie in gaps of the training
distribution? Compare sector-821 training decisions (gravity3L_canon10x raw
jsonl: fields sector_id, target, target_weight) against the 281 truth-path
decisions of rank_diag3.pkl, split good (rank<=20) vs bad (rank>100).

Features per decision TARGET: r, s, n_dots (sum of (x-1) over den slots >1),
den-numerators (negative den slots), ISP powers (nonzero slots 10-14),
max |numerator| power.
"""
import os, sys, json, glob, pickle
from collections import Counter
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
N_DEN = 10


def feats(t):
    r = sum(x for x in t if x > 0)
    s = sum(-x for x in t if x < 0)
    dots = sum(x - 1 for x in t[:N_DEN] if x > 1)
    den_num = sum(1 for x in t[:N_DEN] if x < 0)
    isp = sum(1 for x in t[N_DEN:] if x != 0)
    maxnum = max((-x for x in t if x < 0), default=0)
    return r, s, dots, den_num, isp, maxnum


def summarize(name, targets):
    n = len(targets)
    if n == 0:
        print(f"{name}: EMPTY")
        return
    F = [feats(t) for t in targets]
    print(f"{name}: n={n}")
    for i, lab in enumerate(["r", "s", "dots", "den-numerator slots",
                             "ISP slots used", "max numerator power"]):
        c = Counter(f[i] for f in F)
        tot = sum(c.values())
        top = "  ".join(f"{k}:{100*c[k]//tot}%" for k in sorted(c)[:8])
        print(f"   {lab:22s} {top}")
    both = sum(1 for f in F if f[2] >= 1 and f[1] >= 1)
    s2 = sum(1 for f in F if f[5] >= 2)
    print(f"   dots>=1 AND s>=1: {100*both//n}%   max-numerator>=2: {100*s2//n}%")


# training decisions, sector 821 (sample all workers)
train_821 = []
train_all_sec = Counter()
for f in sorted(glob.glob(ROOT + "/data/gravity3L_canon10x_raw_jsonl/*.jsonl")):
    for line in open(f):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        train_all_sec[d["sector_id"]] += 1
        if d["sector_id"] == 821:
            train_821.append(tuple(d["target"]))

print(f"total training decisions: {sum(train_all_sec.values())} across "
      f"{len(train_all_sec)} sectors; sector 821: {len(train_821)}")

with open(ROOT + "/results/truth/rank_diag3.pkl", "rb") as f:
    diag = pickle.load(f)
good, bad = [], []
for T, rows in diag.items():
    for x in rows:
        if x["rank"] is None:
            continue
        (good if x["rank"] <= 20 else bad if x["rank"] > 100 else []).append(
            tuple(x["J"]))

summarize("\nTRAIN sector-821 targets", train_821)
summarize("\nTRUTH-PATH good (rank<=20) targets", good)
summarize("\nTRUTH-PATH bad (rank>100) targets", bad)

# exact/near coverage: how many bad states appear verbatim in training?
tset = set(train_821)
print(f"\nbad states appearing VERBATIM in sector-821 training targets: "
      f"{sum(1 for t in set(bad) if t in tset)}/{len(set(bad))}")
print(f"good states verbatim: {sum(1 for t in set(good) if t in tset)}"
      f"/{len(set(good))}")
