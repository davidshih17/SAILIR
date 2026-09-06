#!/usr/bin/env python
"""Verify a REGENERATED closure is interchangeable with a shipped one.

Checks the schema the consumers actually rely on:
  beam_search_v9 closure probe -> d['closure'] as [[op, seed], ...]
  batch_closure                -> also writes 'integral', 'sector', 'pivots'
and that seeds are absolute N_IND-length integer vectors (the emitter's
membership test is over absolute seeds, so a relative one would silently
never match).
"""
import glob
import json
import os
import sys

N_IND = 15
REGEN = 'results/truth/closures/regen/out'
SHIPPED = 'results/truth/closures/v5_p101_59k/out'


def summarize(path):
    with open(path) as f:
        d = json.load(f)
    rows = [(int(o), tuple(s)) for o, s in d['closure']]
    return d, rows


def check(path, label):
    d, rows = summarize(path)
    print(f'--- {label}: {os.path.basename(path)}')
    print(f'    keys        : {sorted(d)}')
    print(f'    closure rows: {len(rows)}  unique: {len(set(rows))}')
    ops = sorted({o for o, _ in rows})
    print(f'    op range    : {min(ops)}..{max(ops)}  distinct={len(ops)}')
    bad_len = [s for _, s in rows if len(s) != N_IND]
    bad_typ = [s for _, s in rows if not all(isinstance(x, int) for x in s)]
    print(f'    bad seed len: {len(bad_len)}   non-int seeds: {len(bad_typ)}')
    if 'pivots' in d:
        print(f'    pivots      : {len(d["pivots"])} (parallel: '
              f'{len(d["pivots"]) == len(rows)})')
    print(f'    integral    : {d.get("integral")}  sector={d.get("sector")}')
    return d, rows


def main():
    regen = sorted(glob.glob(os.path.join(REGEN, '*.json')))
    if not regen:
        print('no regenerated closures yet'); return 1
    ship = sorted(glob.glob(os.path.join(SHIPPED, '*.json')))[0]

    ds, rs = check(ship, 'SHIPPED')
    print()
    for p in regen:
        dr, rr = check(p, 'REGEN')
        print()
        same = sorted(ds) == sorted(dr)
        print(f'    key sets identical to shipped: {same}')
        if not same:
            print(f'      only shipped: {sorted(set(ds) - set(dr))}')
            print(f'      only regen  : {sorted(set(dr) - set(ds))}')
        # the target's own integral must be consistent with the filename tag
        tag = os.path.splitext(os.path.basename(p))[0]
        want = [int(x) for x in tag.split('_')]
        print(f'    filename tag matches integral: {want == list(dr["integral"])}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
