#!/usr/bin/env python
"""Paired A/B timing report for two stripped builds.

WHY THIS EXISTS
---------------
Comparing a new build against the ORIGINAL reference run is contaminated: the
reference was measured at a different time, under different cluster load. On
this cluster a single-core job shares memory bandwidth and last-level cache
with whatever else is on the machine, so the same build on the same target has
measured 0.98x and 2.05x hours apart.

Observed directly (2026-09-07): two jobs measuring 1.43x and 1.58x were sharing
node43/node46 with ONE other user holding 28 cores on each box, against my 1.
Same hardware that produced 0.99x when quiet.

So: submit BOTH builds on the SAME targets at the SAME time, and compare them
to EACH OTHER. Shared contention cancels; what is left is the code.

Bit-identicality is immune to all of this, so keep running that over the full
125. Only the TIMING verdict needs pairing, and it needs only a handful of long
targets -- which is why this is cheap (2 x N jobs, not 2 x 125).

USAGE
  ab_timing.py --a DIR --b DIR [--a-label ..] [--b-label ..]
               [--a-cluster N --b-cluster M --targets FILE] [--ref DIR]

--a-cluster/--b-cluster/--targets enable EXECUTE-HOST reporting: the submit
order in FILE gives proc ids, and each userlog names the slot. A pair whose two
jobs landed on different machines is flagged -- the comparison is weaker then,
though still far better than comparing across hours.
"""
import argparse
import glob
import os
import pickle
import re
import statistics as st
import sys


def load(pattern):
    out = {}
    for f in glob.glob(pattern):
        try:
            with open(f, 'rb') as fh:
                out[os.path.basename(f)[:-4]] = pickle.load(fh)
        except Exception:
            pass
    return out


def hosts(logdir, cluster, targets):
    """target -> execute host, via submit order (proc i == line i of targets)."""
    out = {}
    if not (cluster and targets):
        return out
    for i, t in enumerate(targets):
        p = os.path.join(logdir, f'{cluster}_{i}.log')
        if not os.path.exists(p):
            continue
        m = re.findall(r'SlotName:\s*(\S+)@([\w.-]+)', open(p, errors='ignore').read())
        if m:
            out[t] = (m[-1][1].split('.')[0], m[-1][0])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True, help='candidate out/ dir (glob ok)')
    p.add_argument('--b', required=True, help='baseline out/ dir (glob ok)')
    p.add_argument('--a-label', default='A')
    p.add_argument('--b-label', default='B')
    p.add_argument('--ref', help='optional original reference out/ dir, for context only')
    p.add_argument('--a-cluster'), p.add_argument('--b-cluster')
    p.add_argument('--a-logs'), p.add_argument('--b-logs')
    p.add_argument('--targets', help='submit-order target list, for host mapping')
    a = p.parse_args()

    A, B = load(os.path.join(a.a, '*.pkl')), load(os.path.join(a.b, '*.pkl'))
    R = load(os.path.join(a.ref, '*.pkl')) if a.ref else {}
    tg = [l.strip() for l in open(a.targets)] if a.targets else []
    tg = [t for t in tg if t]
    HA = hosts(a.a_logs or '', a.a_cluster, tg)
    HB = hosts(a.b_logs or '', a.b_cluster, tg)

    common = sorted(set(A) & set(B), key=lambda t: -(R.get(t, {}).get('time', 0) or A[t]['time']))
    print(f'=== PAIRED A/B: {a.a_label} vs {a.b_label} ===')
    print(f'{a.a_label} : {a.a}   ({len(A)} targets)')
    print(f'{a.b_label} : {a.b}   ({len(B)} targets)')
    print(f'paired: {len(common)}')
    missing = (set(A) ^ set(B))
    if missing:
        print(f'UNPAIRED (ignored): {len(missing)} -> {", ".join(sorted(missing)[:4])}')
    if not common:
        print('\nnothing paired yet.')
        return 0

    # identical results are the precondition for comparing times at all
    bad = [t for t in common
           if any(A[t].get(f) != B[t].get(f) for f in ('path', 'success', 'final_expr', 'steps'))]
    print(f'\nresults identical between the two builds: {len(common)-len(bad)}/{len(common)}')
    for t in bad[:5]:
        print(f'   DIFFERS: {t}')

    print(f'\n{"target":30s} {a.a_label:>10s} {a.b_label:>10s} {"A/B":>7s}  hosts')
    ratios = []
    for t in common:
        ta, tb = A[t]['time'], B[t]['time']
        ratios.append(ta / tb)
        ha, hb = HA.get(t, ('?', ''))[0], HB.get(t, ('?', ''))[0]
        same = '' if ha == '?' or hb == '?' else ('  same-host' if ha == hb else '  DIFFERENT HOSTS')
        print(f'{t[:30]:30s} {ta:9.1f}s {tb:9.1f}s {ta/tb:6.3f}x  {ha}/{hb}{same}')

    r = sorted(ratios)
    print(f'\n{a.a_label}/{a.b_label}: n={len(r)}  median={st.median(r):.3f}x  '
          f'mean={st.mean(r):.3f}x  min={r[0]:.3f}x  max={r[-1]:.3f}x')
    if R:
        ra = [A[t]['time'] / R[t]['time'] for t in common if t in R]
        rb = [B[t]['time'] / R[t]['time'] for t in common if t in R]
        if ra and rb:
            print(f'  for context vs the ORIGINAL reference (contaminated by load):')
            print(f'    {a.a_label} median {st.median(ra):.3f}x, '
                  f'{a.b_label} median {st.median(rb):.3f}x  '
                  f'-- if BOTH are high, that is the cluster, not the code')
    return 0


if __name__ == '__main__':
    sys.exit(main())
