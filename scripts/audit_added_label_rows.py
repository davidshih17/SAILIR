#!/usr/bin/env python
"""Are the label rows that regeneration ADDED legitimate for that state?

lab grew because the probe set became a UNION: the campaign target's shipped
closure PLUS closures built for other, off-path targets. That is the intended
design -- one closure is the DEPENDENCY closure of its own target, so it covers
the intermediate targets the truth path visits, but NOT ones reached only by
deviating. Still, unioning across targets could in principle admit rows that do
not belong at this state, which would invert the "false positive" reading.

Test: each closure stores `pivots` parallel to `closure` -- the integral each
row was recorded as solving. For every action index regeneration ADDED, we ask
whether that row's pivot matches the state's own target.

CAVEAT, from batch_closure.py's own comment: the pivot is STATE-DEPENDENT --
"the same equation eliminates a different integral depending on the
substitution store". So a mismatch is not proof of a bad row, and a match is
not proof of a good one. It is evidence, not a decision procedure.
"""
import glob
import json
import os
import re
import sys
from collections import Counter

LINE = re.compile(r'^\s*\[(\d+)\]\s+(\S+)\s+->\s+\+(\d+)\s+rows')


def key(d):
    return (tuple(d['target']), d['step'],
            tuple(sorted((tuple(k), v) for k, v in d['expr'])))


def load_phase(y, ph):
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


def main(y):
    # (op, seed) -> set of pivots, and which source it came from
    piv = {}
    src = {}
    files = sorted(glob.glob('results/truth/closures/regen/out/*.json'))
    files += sorted(glob.glob('results/truth/closures/v5_p101_59k/out/*.json'))[:0]
    for p in files:
        with open(p) as f:
            d = json.load(f)
        pv = d.get('pivots')
        for i, (o, sd) in enumerate(d['closure']):
            k = (int(o), tuple(sd))
            src.setdefault(k, os.path.basename(p))
            if pv:
                piv.setdefault(k, set()).add(tuple(pv[i]))
    print(f'regen closure rows indexed: {len(piv):,} from {len(files)} files')

    before, after = load_phase(y, 'before'), load_phase(y, 'after')
    lost = [k for k in before if k not in after]
    print(f'lost rows to audit: {len(lost):,}')
    print()

    stats = Counter()
    examples = []
    for k in lost:
        d, integral = before[k]
        T = d['target']
        N = len(T)
        lab_old = set(d['valid_label_idxs'])
        for j, (o, delta) in enumerate(d['valid_actions']):
            if j in lab_old:
                continue
            seed = tuple(T[i] + delta[i] for i in range(N))
            kk = (int(o), seed)
            if kk not in piv:
                continue                      # not one of the added rows
            stats['added_rows'] += 1
            pivots = piv[kk]
            if tuple(T) in pivots:
                stats['pivot_matches_state_target'] += 1
            else:
                stats['pivot_differs'] += 1
                if len(examples) < 3:
                    examples.append((integral, T, o, seed, sorted(pivots)[:2],
                                     src[kk]))
    print('AUDIT OF ADDED LABEL ROWS')
    for k2, v in stats.most_common():
        print(f'  {k2:28s} {v:8,d}')
    if stats['added_rows']:
        m = stats['pivot_matches_state_target'] / stats['added_rows']
        print(f'  pivot match rate           : {m:.1%}')
    if examples:
        print('\n  examples where the pivot differs from the state target:')
        for integral, T, o, seed, pv, s in examples:
            print(f'    state target : {",".join(str(x) for x in T)}')
            print(f'    row          : op={o} seed={",".join(str(x) for x in seed)}')
            print(f'    its pivot(s) : {pv}')
            print(f'    from closure : {s}')
            print()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
