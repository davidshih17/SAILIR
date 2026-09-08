#!/usr/bin/env python
"""Paired McNemar test on per-target solve outcomes.

Marginal counts (197 vs 205) hide how many targets actually FLIPPED. McNemar
uses only the discordant pairs -- targets one model solves and the other does
not -- which is the correct test for paired binary outcomes on the same targets.
"""
import glob
import os
import re
import sys
from math import comb

PROG = re.compile(r'^\s*\[(\d+)\]\s+(\S+)\s+->')


def verdicts(run):
    """target -> True if SUCCESS, paired by order within each shard."""
    out = {}
    for prog in sorted(glob.glob(os.path.join(run, 'shard_*_progress.log'))):
        shard = os.path.basename(prog).replace('_progress.log', '')
        log = os.path.join(run, f'{shard}.log')
        if not os.path.exists(log):
            continue
        tg = [m.group(2) for m in (PROG.match(l) for l in open(prog)) if m]
        vs = []
        for l in open(log, errors='replace'):
            if l.startswith('SUCCESS'):
                vs.append(True)
            elif l.startswith('INCOMPLETE'):
                vs.append(False)
        for t, v in zip(tg, vs):
            out[t] = v
    return out


def two_sided_binom(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main(a_run, b_run, a_name, b_name):
    A, B = verdicts(a_run), verdicts(b_run)
    common = sorted(set(A) & set(B))
    both = sum(1 for k in common if A[k] and B[k])
    onlyA = sum(1 for k in common if A[k] and not B[k])
    onlyB = sum(1 for k in common if B[k] and not A[k])
    neither = sum(1 for k in common if not A[k] and not B[k])
    print(f'paired targets: {len(common)}')
    print(f'  both solved            : {both}')
    print(f'  only {a_name:<12s}      : {onlyA}   <- lost by {b_name}')
    print(f'  only {b_name:<12s}      : {onlyB}   <- gained by {b_name}')
    print(f'  neither                : {neither}')
    print(f'\n  {a_name} solve rate : {(both+onlyA)/len(common):.1%}')
    print(f'  {b_name} solve rate : {(both+onlyB)/len(common):.1%}')
    p = two_sided_binom(onlyB, onlyA)
    print(f'\nMcNemar (exact, discordant pairs only)')
    print(f'  discordant : {onlyA + onlyB}  ({onlyB} gained, {onlyA} lost)')
    print(f'  two-sided p: {p:.4f}')
    print(f'  verdict    : {"SIGNIFICANT at 0.05" if p < 0.05 else "NOT significant at 0.05"}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
