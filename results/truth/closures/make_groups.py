"""Partition the 18,000 targets into system GROUPS for batched closure builds.

One job per group. Each group shares a single (sector, rmax, smax) system, so
the job builds that system ONCE in memory and reads every one of its targets off
it -- 1,389 builds instead of 18,000.

Grouping rules, both load-bearing:

  1. The rung is the FIRST ADMISSIBLE one in ladder order, which is what
     worker_replay actually picks. Grouping on the CHEAPEST rung instead would
     send jobs to systems the engine never builds, and every job would rebuild.

  2. Groups within a sector are MERGED into the largest box that still fits the
     budget (a bigger box is a strict superset, so it covers them all). Merging
     is capped at the budget precisely so it never creates a system larger than
     the largest we already planned to build: measured worst stays 462,462
     seeds, total seeds rise only 1.07x, builds fall a further 1.67x.

A target may still escalate past its group's rung if no rule is found there;
that is fine and self-correcting -- the job just builds one more system.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from math import comb

B = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
N_DEN, N_IND = 10, 15


def seeds(S, rmax, smax):
    L = bin(S).count('1')
    M = N_IND - L
    return comb(rmax, L) * comb(smax + M, M) if rmax >= L else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default=f'{B}/results/truth/truthcull/g1023_18k.txt')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--budget', type=int, default=500000)
    ap.add_argument('--dr', type=int, default=1)
    ap.add_argument('--ds', type=int, default=1)
    ap.add_argument('--no-merge', action='store_true',
                    help='skip the per-sector merge. Merging raises each '
                         'sector to its MAXIMAL box, which is what drove peak '
                         'RSS to 12-24 GB; un-merged boxes are far smaller '
                         '(median 19,800 seeds ~ 0.8 GB) at 1.67x more builds.')
    args = ap.parse_args()

    rows = [tuple(int(x) for x in ln.split(','))
            for ln in open(args.targets).read().split() if ln.strip()]
    dr, ds = args.dr, args.ds
    rungs = [(dr, ds), (dr+1, ds), (dr+2, ds), (dr+3, ds), (dr+4, ds),
             (dr+5, ds), (dr+1, ds+1), (dr+2, ds+1), (dr+3, ds+1)]

    groups = defaultdict(list)
    blocked = []
    for t in rows:
        r = sum(x for x in t if x > 0)
        s = sum(-x for x in t if x < 0)
        S = 0
        for i in range(N_DEN):
            if t[i] > 0:
                S |= 1 << i
        L = bin(S).count('1')
        ch = None
        for (a, b) in rungs:                    # FIRST admissible, ladder order
            if r + a < L:
                continue
            if seeds(S, r + a, s + b) <= args.budget:
                ch = (r + a, s + b)
                break
        if ch is None:
            blocked.append(t)
            continue
        groups[(S, ch[0], ch[1])].append(t)

    # merge within sector, capped at the budget
    by_sec = defaultdict(list)
    for (S, rm, sm), ts in groups.items():
        by_sec[S].append((rm, sm, ts))
    plan = []
    if args.no_merge:
        for (S, rm, sm), ts in groups.items():
            plan.append((S, rm, sm, ts))
        by_sec = {}
    for S, gs in by_sec.items():
        rm = max(g[0] for g in gs)
        sm = max(g[1] for g in gs)
        if seeds(S, rm, sm) <= args.budget:
            plan.append((S, rm, sm, [t for g in gs for t in g[2]]))
        else:
            for (a, b, ts) in gs:
                plan.append((S, a, b, ts))

    plan.sort(key=lambda p: -seeds(p[0], p[1], p[2]))   # biggest first: long
    #                                                    pole starts earliest
    os.makedirs(args.outdir, exist_ok=True)
    idx = []
    for i, (S, rm, sm, ts) in enumerate(plan):
        p = os.path.join(args.outdir, f'group_{i:05d}.json')
        with open(p, 'w') as f:
            json.dump({'sector': S, 'rmax': rm, 'smax': sm,
                       'seeds': seeds(S, rm, sm),
                       'targets': [list(t) for t in ts]}, f)
        idx.append(p)
    with open(os.path.join(args.outdir, 'groups.txt'), 'w') as f:
        f.write('\n'.join(idx) + '\n')

    ntg = sum(len(p[3]) for p in plan)
    print(f'targets      : {len(rows):,}')
    print(f'blocked      : {len(blocked):,}')
    print(f'GROUPS (jobs): {len(plan):,}')
    print(f'targets in groups: {ntg:,}')
    print(f'total seeds  : {sum(seeds(p[0],p[1],p[2]) for p in plan):,}')
    print(f'worst system : {max(seeds(p[0],p[1],p[2]) for p in plan):,} seeds')
    print(f'targets/group: mean {ntg/len(plan):.1f}  '
          f'max {max(len(p[3]) for p in plan)}')
    print(f'wrote {len(idx):,} group files -> {args.outdir}')
    assert ntg == len(rows) - len(blocked), 'target accounting mismatch'
    return 0


if __name__ == '__main__':
    sys.exit(main())
