#!/usr/bin/env python
"""Aggregate a parallel DAgger round's per-shard logs.

Reads the [DAGGER] summary line each collector process prints at exit and the
[CLOSUREPROBE] per-step lines, and reports the numbers that decide whether
regeneration is worth its CPU:

  exhaustion rate   deadend / seen      -- how often the expert has nothing
  targets needing a closure             -- the regen work list size
  regen per target  work list / targets -- what a round costs to label fully
"""
import glob
import os
import re
import sys
from collections import Counter

DAG = re.compile(
    r"\[DAGGER\] seen=(\d+) emitted=(\d+) skipped_model_already_right=(\d+) "
    r"deadend=(\d+)")
PROBE = re.compile(
    r"\[CLOSUREPROBE\] step=\d+ tasks=(\d+) expert_can_label=(\d+) "
    r"exhausted=(\d+)")


def main(run):
    tot = Counter()
    n_summaries = 0
    for f in sorted(glob.glob(os.path.join(run, 'shard_*.log'))):
        for ln in open(f, errors='replace'):
            m = DAG.search(ln)
            if m:
                seen, emit, skip, dead = (int(x) for x in m.groups())
                tot['seen'] += seen; tot['emitted'] += emit
                tot['skipped'] += skip; tot['deadend'] += dead
                n_summaries += 1
                continue
            m = PROBE.search(ln)
            if m:
                tasks, canlab, exh = (int(x) for x in m.groups())
                tot['p_tasks'] += tasks; tot['p_canlab'] += canlab
                tot['p_exh'] += exh

    tgt_f = os.path.join(run, 'targets.txt')
    miss_f = os.path.join(run, 'missing_targets.txt')
    n_tgt = sum(1 for _ in open(tgt_f)) if os.path.exists(tgt_f) else 0
    n_miss = sum(1 for _ in open(miss_f)) if os.path.exists(miss_f) else 0

    seen = tot['seen'] or 1
    print(f"  collector summaries parsed : {n_summaries}")
    print(f"  states visited (seen)      : {tot['seen']:,}")
    print(f"    emitted as training rows : {tot['emitted']:,} "
          f"({tot['emitted']/seen:.1%})")
    print(f"    model already right      : {tot['skipped']:,} "
          f"({tot['skipped']/seen:.1%})")
    print(f"    dead ends (no expert)    : {tot['deadend']:,} "
          f"({tot['deadend']/seen:.1%})")
    if tot['p_tasks']:
        pt = tot['p_tasks']
        print(f"  probe tasks                : {pt:,}  "
              f"labelable {tot['p_canlab']/pt:.1%}  "
              f"exhausted {tot['p_exh']/pt:.1%}")
    print(f"  campaign targets           : {n_tgt:,}")
    print(f"  targets needing a closure  : {n_miss:,}")
    if n_tgt:
        print(f"  regen closures per target  : {n_miss/n_tgt:.3f}")
        # cost at the production mean of 3.41 CPU-min per closure
        print(f"  projected regen cost       : {n_miss*3.41/60:.1f} CPU-h "
              f"(at the production 3.41 CPU-min/closure)")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else '.'))
