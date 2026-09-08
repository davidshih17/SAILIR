#!/usr/bin/env python
"""Paired per-target comparison of two checkpoints on the SAME targets.

Both arms walked the identical 300 held-out targets with identical settings, so
the comparison can be PAIRED per target rather than pooled -- which is the
honest analysis, since states within one rollout are correlated and pooling
over-states significance.

Metric: top1 = already_right / (already_right + emitted), over labelable
states. Reported alongside the dead-end share, because a policy that merely
wanders somewhere different could raise one while worsening the other.
"""
import glob
import os
import re
import sys
from statistics import mean, stdev

DAG = re.compile(r"\[DAGGER\] seen=(\d+) emitted=(\d+) "
                 r"skipped_model_already_right=(\d+) deadend=(\d+)")
PROG = re.compile(r'^\s*\[(\d+)\]\s+(\S+)\s+->')


def per_target(run):
    """target -> (seen, emitted, right, dead), paired by order within a shard."""
    out = {}
    for prog in sorted(glob.glob(os.path.join(run, 'shard_*_progress.log'))):
        shard = os.path.basename(prog).replace('_progress.log', '')
        log = os.path.join(run, f'{shard}.log')
        if not os.path.exists(log):
            continue
        tgts = [m.group(2) for m in (PROG.match(l) for l in open(prog)) if m]
        summ = [tuple(int(x) for x in m.groups())
                for m in (DAG.search(l) for l in open(log, errors='replace')) if m]
        for t, s in zip(tgts, summ):
            out[t] = s
    return out


def main(a_run, b_run, a_name, b_name):
    A, B = per_target(a_run), per_target(b_run)
    common = sorted(set(A) & set(B))
    print(f'{a_name}: {len(A)} targets   {b_name}: {len(B)} targets   '
          f'paired: {len(common)}')

    def pooled(D, keys):
        se = sum(D[k][0] for k in keys); em = sum(D[k][1] for k in keys)
        ri = sum(D[k][2] for k in keys); de = sum(D[k][3] for k in keys)
        lab = em + ri
        return se, em, ri, de, ri / lab if lab else 0.0, de / se if se else 0.0

    for name, D in ((a_name, A), (b_name, B)):
        se, em, ri, de, top1, dead = pooled(D, common)
        print(f'\n{name}')
        print(f'  states visited   : {se:,}')
        print(f'  top-1 correct    : {ri:,} / {em+ri:,} = {top1:.2%}')
        print(f'  emitted (errors) : {em:,}')
        print(f'  dead ends        : {de:,} ({dead:.1%})')

    # paired per-target differences
    diffs, wins, losses, ties = [], 0, 0, 0
    for k in common:
        la = A[k][1] + A[k][2]; lb = B[k][1] + B[k][2]
        if la == 0 or lb == 0:
            continue
        da = A[k][2] / la; db = B[k][2] / lb
        d = db - da
        diffs.append(d)
        if d > 1e-9: wins += 1
        elif d < -1e-9: losses += 1
        else: ties += 1
    m = mean(diffs); s = stdev(diffs) if len(diffs) > 1 else 0.0
    se_m = s / (len(diffs) ** 0.5)
    print(f'\nPAIRED per-target change in top-1 rate ({b_name} - {a_name})')
    print(f'  targets compared : {len(diffs)}')
    print(f'  mean change      : {m:+.2%}')
    print(f'  std / SE         : {s:.2%} / {se_m:.2%}')
    print(f'  95% CI           : [{m-1.96*se_m:+.2%}, {m+1.96*se_m:+.2%}]')
    print(f'  targets better   : {wins}   worse: {losses}   tied: {ties}')
    if se_m > 0:
        print(f'  t-statistic      : {m/se_m:.1f}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
