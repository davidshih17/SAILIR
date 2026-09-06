#!/usr/bin/env python
"""Split a DAgger round's merged rows.jsonl into per-trajectory rec_<tag>.jsonl.

Two things the packer needs that a single merged jsonl cannot give it:

1. ONE FILE PER TRAJECTORY. pack_shard.py assigns shards by crc32 of the
   FILENAME and splits train/val by file, precisely so a trajectory never
   straddles the split (TRAIN_FROM_SCRATCH.md: 756 of 890 trajectories in one
   shard were missing steps that had landed in val -- leakage, steps from one
   reduction on both sides). A merged file would put all 300 rollouts in one
   unit; splitting by the per-step `target` instead would scatter one rollout
   across 4,170 files, which is the same leakage in the other direction.

2. A CORRECT start_target. preprocess_to_tensors.py reads start_target from the
   FIRST LINE of a file and applies it to every sample in it. The emitter wrote
   the per-task target into that field, so it is only right by accident. Here we
   set it explicitly to the campaign integral T, which is what the field means:
   "the whole task is defined relative to T ... T is the floor of the active
   region". Harmless today (use_start_target defaults off) but it would
   silently corrupt any future EXP 1 run.

The row->target mapping is recovered from each shard's progress log, whose
cumulative counts are exact (verified: 1546 logged == 1546 lines).
"""
import argparse
import glob
import json
import os
import re

LINE = re.compile(r'^\s*\[(\d+)\]\s+(\S+)\s+->\s+\+(\d+)\s+rows\s+\(total\s+(\d+)\)')


def tag_of(vec):
    # MUST match batch_closure.tag_of / worker.sh: ',' -> '_', minus PRESERVED
    return '_'.join(str(int(x)) for x in vec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='a results/dagger/round_* dir')
    ap.add_argument('--outdir', required=True, help='where rec_*.jsonl go')
    ap.add_argument('--fix-start-target', action='store_true', default=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    n_files = n_rows = n_empty = 0
    mismatch = []
    for prog in sorted(glob.glob(os.path.join(args.run, 'shard_*_progress.log'))):
        shard = os.path.basename(prog).replace('_progress.log', '')
        rows_f = os.path.join(args.run, f'{shard}_rows.jsonl')
        if not os.path.exists(rows_f):
            continue
        with open(rows_f) as f:
            rows = f.readlines()

        plan = []
        for ln in open(prog):
            m = LINE.match(ln)
            if m:
                plan.append((m.group(2), int(m.group(3)), int(m.group(4))))
        if plan and plan[-1][2] != len(rows):
            mismatch.append((shard, plan[-1][2], len(rows)))
            continue

        pos = 0
        for integral, cnt, _cum in plan:
            chunk = rows[pos:pos + cnt]
            pos += cnt
            if not chunk:
                n_empty += 1
                continue
            T = [int(x) for x in integral.split(',')]
            out = os.path.join(args.outdir, f'rec_{tag_of(T)}.jsonl')
            with open(out, 'w') as g:
                for r in chunk:
                    d = json.loads(r)
                    if args.fix_start_target:
                        d['start_target'] = T
                    g.write(json.dumps(d) + '\n')
            n_files += 1
            n_rows += len(chunk)

    print(f'trajectory files written : {n_files:,}')
    print(f'rows written             : {n_rows:,}')
    print(f'targets with zero rows   : {n_empty:,}  (model was right everywhere)')
    if mismatch:
        print('MISMATCHED SHARDS (skipped, log total != actual lines):')
        for s, a, b in mismatch:
            print(f'  {s}: log={a} actual={b}')
    else:
        print('all shards reconciled exactly')


if __name__ == '__main__':
    main()
