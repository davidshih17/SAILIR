#!/usr/bin/env python
"""Verify the packed DAgger shards match the base corpus's format.

Checks the things that would silently break training:
  - same tensor keys as data/shard_*/train.pt (the loader indexes by key)
  - sample counts reconcile against the emitted rows
  - labels present and non-empty (BCE needs label_mask)
  - coefficients within GF(101), matching --prime 101
"""
import glob
import os
import sys
import torch


def summarize(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    return d


def main():
    base = sorted(glob.glob('data/shard_*/train.pt'))[0]
    dag = sorted(glob.glob('data_dagger/shard_*/train.pt'))
    print(f'base shard : {base}')
    print(f'dagger shards: {len(dag)}')

    b = summarize(base)
    a = summarize(dag[0])
    kb, ka = set(b), set(a)
    print(f'\nkeys base   : {len(kb)}')
    print(f'keys dagger : {len(ka)}')
    print(f'  missing in dagger : {sorted(kb - ka) or "none"}')
    print(f'  extra in dagger   : {sorted(ka - kb) or "none"}')

    n_tr = n_va = 0
    maxc = 0
    lab_empty = 0
    lab_total = 0
    for p in dag:
        d = summarize(p)
        n = len(d['target_integrals'])
        n_tr += n
        c = d.get('expr_coeffs')
        if c is not None and len(c):
            maxc = max(maxc, int(c.abs().max()))
        # CSR layout: flat valid_label_idxs + valid_label_offsets boundaries
        off = d.get('valid_label_offsets')
        if off is not None:
            cnt = (off[1:] - off[:-1])
            lab_total += int(cnt.numel())
            lab_empty += int((cnt == 0).sum())
            globals()['LAB_SUM'] = globals().get('LAB_SUM', 0) + int(cnt.sum())
        v = p.replace('train.pt', 'val.pt')
        if os.path.exists(v):
            n_va += len(summarize(v)['target_integrals'])
    print(f'\ntrain samples : {n_tr:,}')
    print(f'val   samples : {n_va:,}')
    print(f'total         : {n_tr + n_va:,}   (emitted rows were 131,738)')
    print(f'max |coeff|   : {maxc}   (must be < 101 for --prime 101)')
    if lab_total:
        print(f'label lists   : {lab_total:,}  empty: {lab_empty:,} '
              f'({lab_empty/lab_total:.2%})')
        print(f'mean correct actions/sample : '
              f'{globals().get("LAB_SUM",0)/lab_total:.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
