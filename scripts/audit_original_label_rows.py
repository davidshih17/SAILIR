#!/usr/bin/env python
"""Baseline: do the ORIGINAL (shipped-closure) labels have matching pivots?

If lab_old rows overwhelmingly have pivot == the state's target, then the
emitter's membership test was only ever meant to be evaluated against a closure
whose rows solve THAT target -- and the cross-target union I introduced is
admitting rows that do not, inflating lab and wrongly reclassifying states as
"model already right".
"""
import glob
import json
import os
import re
import sys
from collections import Counter

LINE = re.compile(r'^\s*\[(\d+)\]\s+(\S+)\s+->\s+\+(\d+)\s+rows')


def load_phase(y, ph):
    out = []
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
                out.append((d, integral))
            pos += cnt
    return out


def shipped_for(tag):
    for D in ('results/truth/closures/v5_p101_59k/out',
              'results/truth/closures/v4_p101_18k/out',
              'results/truth/closures/v5_p101_59k/retry',
              'results/truth/closures/v4_p101_18k/retry'):
        p = os.path.join(D, f'{tag}.json')
        if os.path.exists(p):
            return p
    return None


def index(paths):
    piv = {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        pv = d.get('pivots')
        if not pv:
            continue
        for i, (o, sd) in enumerate(d['closure']):
            piv.setdefault((int(o), tuple(sd)), set()).add(tuple(pv[i]))
    return piv


def main(y):
    rows = load_phase(y, 'before')
    print(f'phase-A rows: {len(rows):,}')
    cache = {}
    st = Counter()
    for d, integral in rows:
        tag = '_'.join(str(int(x)) for x in
                       (int(v) for v in integral.split(',')))
        if tag not in cache:
            p = shipped_for(tag)
            cache[tag] = index([p]) if p else {}
        piv = cache[tag]
        T = d['target']; N = len(T)
        for j in d['valid_label_idxs']:
            o, delta = d['valid_actions'][j]
            seed = tuple(T[i] + delta[i] for i in range(N))
            k = (int(o), seed)
            if k not in piv:
                st['label_row_not_in_shipped_closure'] += 1
                continue
            st['label_rows'] += 1
            if tuple(T) in piv[k]:
                st['pivot_matches_state_target'] += 1
            else:
                st['pivot_differs'] += 1
    print('AUDIT OF ORIGINAL (lab_old) LABEL ROWS')
    for k, v in st.most_common():
        print(f'  {k:36s} {v:8,d}')
    if st['label_rows']:
        print(f"  pivot match rate                     : "
              f"{st['pivot_matches_state_target']/st['label_rows']:.1%}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
