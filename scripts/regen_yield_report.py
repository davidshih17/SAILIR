#!/usr/bin/env python
"""Report what regeneration bought, from a yield test's before/after phases.

The rollout is deterministic given the model, so `seen` must match between the
phases; if it does not, something other than the closure library changed and
the comparison is void. Everything else is a straight difference.
"""
import glob
import os
import re
import sys
from collections import Counter

DAG = re.compile(
    r"\[DAGGER\] seen=(\d+) emitted=(\d+) skipped_model_already_right=(\d+) "
    r"deadend=(\d+)")


def phase(y, name):
    t = Counter()
    for f in sorted(glob.glob(os.path.join(y, name, 'shard_*.log'))):
        for ln in open(f, errors='replace'):
            m = DAG.search(ln)
            if m:
                s, e, k, d = (int(x) for x in m.groups())
                t['seen'] += s; t['emit'] += e; t['skip'] += k; t['dead'] += d
    rows = os.path.join(y, name, 'rows.jsonl')
    t['rows'] = sum(1 for _ in open(rows)) if os.path.exists(rows) else 0
    miss = os.path.join(y, name, 'missing.txt')
    t['miss'] = sum(1 for _ in open(miss)) if os.path.exists(miss) else 0
    return t


def main(y, n_regen):
    a, b = phase(y, 'before'), phase(y, 'after')
    n_regen = int(n_regen)

    print(f"{'':26s} {'before':>10s} {'after':>10s} {'delta':>10s}")
    for k, lab in [('seen', 'states visited'), ('emit', 'emitted rows'),
                   ('skip', 'model already right'), ('dead', 'dead ends'),
                   ('miss', 'exhausted targets')]:
        print(f"  {lab:24s} {a[k]:10,d} {b[k]:10,d} {b[k]-a[k]:+10,d}")
    print(f"  {'rows.jsonl lines':24s} {a['rows']:10,d} {b['rows']:10,d} "
          f"{b['rows']-a['rows']:+10,d}")

    print()
    if a['seen'] != b['seen']:
        print(f"  WARNING: seen changed ({a['seen']} -> {b['seen']}). The "
              f"rollout should be deterministic; comparison is NOT clean.")
    recovered = a['dead'] - b['dead']
    new_rows = b['rows'] - a['rows']
    print(f"  closures regenerated      : {n_regen:,}")
    print(f"  dead ends recovered       : {recovered:,}")
    print(f"  NEW TRAINING ROWS         : {new_rows:,}")
    if recovered > 0:
        print(f"  rows per recovered state  : {new_rows/recovered:.3f}")
        print(f"    (the round's labelable states are 49% model-wrong, so ~0.49"
              f" would mean dead ends behave like ordinary states; ~0 means "
              f"they are states the model already had right)")
    if n_regen > 0:
        print(f"  rows per closure built    : {new_rows/n_regen:.3f}")
        cost_h = n_regen * 3.41 / 60
        print(f"  cost of this test         : {cost_h:.1f} CPU-h "
              f"(at 3.41 CPU-min/closure)")
        if new_rows > 0:
            print(f"  CPU-h per 1,000 new rows  : "
                  f"{cost_h/new_rows*1000:.1f}")
        else:
            print(f"  CPU-h per 1,000 new rows  : INFINITE (zero yield)")
    print()
    print(f"  EXTRAPOLATION to the full 1,461-target work list:")
    if n_regen > 0:
        print(f"    projected new rows      : {new_rows/n_regen*1461:,.0f}")
        print(f"    projected cost          : {1461*3.41/60:.0f} CPU-h")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 0))
