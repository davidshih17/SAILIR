#!/usr/bin/env python
"""Show concretely what the 116 disappeared rows are.

A row is emitted only when  lab != []  AND  model_top1 NOT IN lab,
where lab = this state's legal actions that are unused closure rows.
It is skipped as "already right" when model_top1 IN lab.

Regeneration only ever ADDS closure rows, so lab can only grow. A state that
was emitted before and is absent after therefore satisfies:

    top1 NOT IN lab_old      (it was emitted)
    top1     IN lab_new      (it is now skipped)
  =>  top1 IN lab_new \\ lab_old

i.e. the model's chosen action was ALWAYS correct, and the old closure library
simply did not contain the row proving it. The row we wrote recorded
valid_label_idxs = lab_old, a "correct set" that EXCLUDED the model's correct
action -- so training on it teaches the model that a right move is wrong.

We do not need to re-run the model to establish this: states are conserved and
the rollout is deterministic, so set difference is proof. What this script adds
is the concrete arithmetic per row -- lab_old vs lab_new on the same state.
"""
import glob
import json
import os
import re
import sys

LINE = re.compile(r'^\s*\[(\d+)\]\s+(\S+)\s+->\s+\+(\d+)\s+rows')


def key(d):
    return (tuple(d['target']), d['step'],
            tuple(sorted((tuple(k), v) for k, v in d['expr'])))


def load_phase(y, ph):
    """rows keyed by state, plus the campaign target each row came from."""
    out = {}
    for prog in sorted(glob.glob(os.path.join(y, ph, 'shard_*_progress.log'))):
        shard = os.path.basename(prog).replace('_progress.log', '')
        rf = os.path.join(y, ph, f'{shard}_rows.jsonl')
        if not os.path.exists(rf):
            continue
        rows = [json.loads(l) for l in open(rf)]
        plan = [(m.group(2), int(m.group(3)))
                for m in (LINE.match(l) for l in open(prog)) if m]
        pos = 0
        for integral, cnt in plan:
            for d in rows[pos:pos + cnt]:
                out[key(d)] = (d, integral)
            pos += cnt
    return out


def closure_set(paths):
    s = set()
    for p in paths:
        with open(p) as f:
            s.update((int(o), tuple(sd)) for o, sd in json.load(f)['closure'])
    return s


def shipped_for(tag):
    for D in ('results/truth/closures/v5_p101_59k/out',
              'results/truth/closures/v4_p101_18k/out',
              'results/truth/closures/v5_p101_59k/retry',
              'results/truth/closures/v4_p101_18k/retry'):
        p = os.path.join(D, f'{tag}.json')
        if os.path.exists(p):
            return p
    return None


def main(y, n_show=3):
    before, after = load_phase(y, 'before'), load_phase(y, 'after')
    lost = [k for k in before if k not in after]
    gained = [k for k in after if k not in before]
    print(f'rows before {len(before):,}   after {len(after):,}')
    print(f'DISAPPEARED {len(lost):,}   appeared {len(gained):,}')
    print()

    regen = sorted(glob.glob('results/truth/closures/regen/out/*.json'))
    new_rows = closure_set(regen)
    print(f'regenerated closure rows available: {len(new_rows):,} '
          f'from {len(regen)} files')
    print()

    shown = 0
    for k in lost:
        d, integral = before[k]
        tag = '_'.join(str(int(x)) for x in
                       (int(v) for v in integral.split(',')))
        sp = shipped_for(tag)
        old = closure_set([sp]) if sp else set()
        both = old | new_rows

        N = len(d['target'])
        T = d['target']
        lab_old = set(d['valid_label_idxs'])
        lab_new = set()
        for j, (o, delta) in enumerate(d['valid_actions']):
            seed = tuple(T[i] + delta[i] for i in range(N))
            if (int(o), seed) in both:
                lab_new.add(j)

        added = sorted(lab_new - lab_old)
        if not added:
            continue
        shown += 1
        print(f'--- lost row {shown} ---')
        print(f'  campaign target      : {integral}')
        print(f'  state target         : {",".join(str(x) for x in T)}  step={d["step"]}')
        print(f'  legal actions        : {d["num_valid_actions"]}')
        print(f'  lab_old (what we wrote as the CORRECT set) : '
              f'{sorted(lab_old)}  ({len(lab_old)})')
        print(f'  lab_new (with regenerated closure rows)    : '
              f'{sorted(lab_new)}  ({len(lab_new)})')
        print(f'  action indices the OLD library missed      : {added}')
        print(f'  => the model picked one of {added}: correct all along, but')
        print(f'     the row we emitted listed only {sorted(lab_old)} as correct.')
        print()
        if shown >= n_show:
            break
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3))
