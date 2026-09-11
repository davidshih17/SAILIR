#!/usr/bin/env python
"""Training-data coverage for the 2+ day long-runner integrals:
per affected sector — decision counts in gravity3L_canon10x, verbatim
membership of the stuck states, and how the stuck (dots, s) classes sit in
that sector's training marginals. Reads /tmp/longrunners.txt."""
import json
import glob
from collections import Counter, defaultdict

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
N_DEN = 10

stuck = []
for ln in open("/tmp/longrunners.txt"):
    ln = ln.strip()
    if ln:
        stuck.append(tuple(int(x) for x in ln.split(",")))


def sec(t):
    return sum(1 << i for i in range(N_DEN) if t[i] > 0)


def cls(t):
    dots = sum(x - 1 for x in t[:N_DEN] if x > 1)
    s = sum(-x for x in t if x < 0)
    return dots, s


sectors = sorted({sec(t) for t in stuck})
stuck_by_sec = defaultdict(list)
for t in stuck:
    stuck_by_sec[sec(t)].append(t)

# v1 restrict list (training) — which of our sectors were even scrambled?
v1 = {int(x) for x in open(ROOT + "/results/canonical_sectors_GR.txt")
      .read().replace("\n", ",").split(",") if x.strip()}
print("stuck sectors missing from the TRAINING restrict list: "
      f"{[S for S in sectors if S not in v1] or 'NONE'}")

count = Counter()
verbatim = set()
cls_train = defaultdict(Counter)
stuck_set = set(stuck)
for f in sorted(glob.glob(ROOT + "/data/gravity3L_canon10x_raw_jsonl/*.jsonl")):
    for line in open(f):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        S = d["sector_id"]
        if S not in stuck_by_sec:
            continue
        count[S] += 1
        tg = tuple(d["target"])
        if tg in stuck_set:
            verbatim.add(tg)
        cls_train[S][cls(tg)] += 1

print(f"\n{'sector':>7} {'#stuck':>6} {'train-decisions':>15}  "
      f"stuck-class coverage (train decisions in that (dots,s) class)")
for S in sectors:
    classes = Counter(cls(t) for t in stuck_by_sec[S])
    parts = []
    for c, n in sorted(classes.items()):
        parts.append(f"(d{c[0]},s{c[1]})x{n}: {cls_train[S].get(c, 0)}")
    print(f"{S:>7} {len(stuck_by_sec[S]):>6} {count[S]:>15}  " + "  ".join(parts))
print(f"\nstuck integrals appearing VERBATIM as training targets: "
      f"{len(verbatim)}/{len(stuck)}")
