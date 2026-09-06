#!/usr/bin/env python
"""Sample training targets from the REAL gravity3L reduction (g1023).

WHY NOT SYNTHESISE. The existing 1848-target corpus is dots-only: every start
has indices in [0,6] and s = 0, i.e. NO NUMERATORS, EVER. The real reduction is
the opposite -- 90.1% of its integrals carry negative indices and s has median
2. So the deployed model has never seen the dominant feature of the integrals it
is asked to reduce.

Uniform sampling over an index box would be worse, not better: uniform over
[-8,8]^15 gives sum|a| ~ 64 against the real median of 11, puts positive values
in the ISP slots (positions 10-14 are <= 0 in ALL 4000 sampled real integrals),
and destroys the sparsity (median 8 nonzero of 15) that IS the sector structure.

Sampling the real integrals is on-distribution by construction and needs no
fitted marginals -- which would in any case break the correlations between
positions that a sector imposes.

PROVENANCE, verified before use (2000-file random sample):
    success=True on 2000/2000, 0 unreadable
    filename parse agrees with `original_integral` on 2000/2000
so every sampled target is one this reduction actually solved.

  positions 0-9  denominators, mean +0.34..+1.64, negative 5-13% of the time
  positions 10-14 ISPs, max 0 everywhere, zero 84-87% of the time
  nonzero indices median 8 of 15;  sum|a| median 11, max 14
"""
import argparse
import os
import pickle
import random
import re
import sys

G = ('/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
     '/results/gr_reduce/g1023/work/results')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=18000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20260902)
    ap.add_argument('--results', default=G)
    ap.add_argument('--exclude', action='append',
                    help='file(s) of integrals to EXCLUDE (comma form, one per '
                         'line; extra whitespace-separated columns ignored). '
                         'Repeatable. Use for BOTH an existing corpus and any '
                         'held-out set.')
    ap.add_argument('--verify', type=int, default=200,
                    help='reopen this many sampled pkls and confirm success')
    args = ap.parse_args()

    names = [n for n in os.listdir(args.results) if n.endswith('.pkl')]
    print(f'result files: {len(names):,}', flush=True)

    # The integral is the filename minus the leading BATCH ID:
    #   async_<batch>_<i0>_..._<i14>.pkl
    # Verified against each pkl's own `original_integral` field.
    seen = {}
    for n in names:
        m = re.match(r'async_(-?\d+(?:_-?\d+)*)\.pkl$', n)
        if not m:
            continue
        v = [int(x) for x in m.group(1).split('_')]
        if len(v) != 16:
            continue
        seen.setdefault(tuple(v[1:]), n)
    print(f'distinct integrals: {len(seen):,}', flush=True)

    # --exclude: drop integrals already used, so a second draw EXTENDS the first
    # instead of overlapping it. Re-running with a larger --n does NOT give a
    # superset -- rng.sample reshuffles for every n -- so extending a corpus
    # requires excluding explicitly.
    #
    # Exclude BOTH the targets already in a corpus AND any held-out set. A
    # held-out target that reappears in training silently destroys the only
    # clean measurement we have on it.
    excl = set()
    for p in (args.exclude or []):
        with open(p) as fh:
            for line in fh:
                line = line.strip().split()[0] if line.strip() else ''
                if line:
                    excl.add(tuple(int(x) for x in line.split(',')))
    if excl:
        before = len(seen)
        seen = {k: v for k, v in seen.items() if k not in excl}
        print(f'excluded {before - len(seen):,} of {len(excl):,} listed '
              f'-> {len(seen):,} available', flush=True)

    if args.n > len(seen):
        raise SystemExit(f'asked for {args.n:,} but only {len(seen):,} distinct')

    rng = random.Random(args.seed)
    keys = sorted(seen)                      # sort first: os.listdir order is
    pick = rng.sample(keys, args.n)          # arbitrary, so seeding alone would
                                             # not make this reproducible
    # VERIFY A SAMPLE really was solved -- the whole point is to generate truth
    # reductions for these, so a target the original run failed on is not a
    # target we know is reachable.
    bad = 0
    for k in rng.sample(pick, min(args.verify, len(pick))):
        with open(os.path.join(args.results, seen[k]), 'rb') as fh:
            d = pickle.load(fh)
        if not d.get('success') or tuple(d.get('original_integral', ())) != k:
            bad += 1
    print(f'verified {min(args.verify, len(pick))} sampled targets: '
          f'{bad} failed the success/identity check', flush=True)
    if bad:
        raise SystemExit('sampled targets include unsolved or mislabelled rows')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        for k in pick:
            f.write(','.join(str(x) for x in k) + '\n')

    import numpy as np
    A = np.array(pick)
    r = np.clip(A, 0, None).sum(1)
    s = np.clip(-A, 0, None).sum(1)
    print(f'\nwrote {len(pick):,} targets -> {args.out}')
    print(f'  index range      : [{A.min()}, {A.max()}]')
    print(f'  with negatives   : {100*(A < 0).any(1).mean():.1f}%')
    print(f'  r  median {int(np.median(r))}  p90 {int(np.percentile(r, 90))}  max {r.max()}')
    print(f'  s  median {int(np.median(s))}  p90 {int(np.percentile(s, 90))}  max {s.max()}')
    print(f'  nonzero idx median {int(np.median((A != 0).sum(1)))} of {A.shape[1]}')


if __name__ == '__main__':
    sys.exit(main())
