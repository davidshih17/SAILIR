#!/usr/bin/env python
"""Report a stripped greedy build against the reference run.

Reports facts, passes no judgement: bit-identicality, the timing DISTRIBUTION,
and peak memory. No pass/fail stamp and no tolerances -- you read the numbers
and decide.

Works on a partial run: everything is computed over whatever has finished, and
the targets still outstanding are listed with their reference times, because
the slowest jobs finish last and a partial set is biased toward the easy ones.

Timing is never a single number:
  agg    = sum(new)/sum(ref)  -- duration-weighted, dominated by the big targets
  mean   = unweighted mean of per-target ratios
  median = median of per-target ratios
  min/max= the extremes, where a real regression shows up first
Short targets are broken out separately rather than dropped: on a few-second
job, startup and node speed swamp the search, and ratios there run 0.19x-4.2x
on builds that are provably bit-identical.

Usage:
  certify_strip.py --ref DIR --new DIR [--long-sec 120] [--label NAME]
"""
import argparse
import glob
import os
import pickle
import statistics as st
import sys

FIELDS = ('path', 'success', 'final_expr', 'steps')


def load(pattern):
    out = {}
    for f in glob.glob(pattern):
        try:
            with open(f, 'rb') as fh:
                out[os.path.basename(f)[:-4]] = pickle.load(fh)
        except Exception as e:                      # truncated/partial pickle
            print(f'  [warn] unreadable {f}: {e}')
    return out


def dist(rows):
    """rows = [(target, ref_sec, new_sec)] -> summary dict, or None if empty."""
    if not rows:
        return None
    r = sorted(b / a for _, a, b in rows if a > 0)
    ra = sum(a for _, a, _ in rows)
    rb = sum(b for _, _, b in rows)
    return dict(n=len(rows), agg=rb / ra, mean=st.mean(r), median=st.median(r),
                lo=r[0], hi=r[-1], ref_tot=ra, new_tot=rb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', required=True, help='reference out/ dir (glob ok)')
    p.add_argument('--new', required=True, help='candidate out/ dir (glob ok)')
    p.add_argument('--long-sec', type=float, default=120.0,
                   help='ref-seconds threshold for the search-dominated bucket')
    p.add_argument('--label', default='candidate')
    a = p.parse_args()

    ref = load(os.path.join(a.ref, '*.pkl'))
    new = load(os.path.join(a.new, '*.pkl'))
    common = sorted(set(ref) & set(new))
    missing = sorted(set(ref) - set(new), key=lambda t: -ref[t]['time'])

    print(f'=== {a.label} ===')
    print(f'reference : {a.ref}  ({len(ref)} targets)')
    print(f'candidate : {a.new}  ({len(new)} targets)')
    print(f'compared  : {len(common)}'
          + (f'   OUTSTANDING: {len(missing)}' if missing else '   (complete)'))
    if missing:
        print('  still running (ref time, longest first):')
        for t in missing[:8]:
            print(f'    {t[:44]:44s} ref={ref[t]["time"]:7.1f}s')
        if len(missing) > 8:
            print(f'    ... and {len(missing)-8} more')
    if not common:
        print('\nnothing to compare yet.')
        return 0

    # ---- bit-identicality ----------------------------------------------------
    bad = {}
    for t in common:
        diff = [f for f in FIELDS if ref[t].get(f) != new[t].get(f)]
        if diff:
            bad[t] = diff
    print(f'\n[1] bit-identical : {len(common)-len(bad)}/{len(common)}'
          f'   ({", ".join(FIELDS)})')
    for t, d in list(bad.items())[:10]:
        print(f'      MISMATCH {t}: differs in {d}')

    # ---- timing --------------------------------------------------------------
    rows = [(t, ref[t]['time'], new[t]['time']) for t in common]
    print('\n[2] timing ratio new/ref, by reference duration'
          + ('   [FINISHED TARGETS ONLY]' if missing else ''))
    print(f'    {"bucket":>10s} {"n":>4s} {"agg":>8s} {"mean":>8s} '
          f'{"median":>8s} {"min":>8s} {"max":>8s} {"ref_tot":>10s}')
    long_lab = f'>={int(a.long_sec)}s'
    for name, lo, hi in [('<30s', 0.0, 30.0), ('30-%ds' % a.long_sec, 30.0, a.long_sec),
                         (long_lab, a.long_sec, float('inf')), ('ALL', 0.0, float('inf'))]:
        d = dist([r for r in rows if lo <= r[1] < hi])
        if d:
            print(f'    {name:>10s} {d["n"]:4d} {d["agg"]:7.3f}x {d["mean"]:7.3f}x '
                  f'{d["median"]:7.3f}x {d["lo"]:7.3f}x {d["hi"]:7.3f}x '
                  f'{d["ref_tot"]:9.0f}s')

    longs = sorted([r for r in rows if r[1] >= a.long_sec], key=lambda r: -r[1])
    if longs:
        print(f'\n    every {long_lab} target, longest first:')
        for t, x, y in longs:
            print(f'      {t[:44]:44s} ref={x:7.1f}s new={y:7.1f}s {y/x:6.3f}x')
    else:
        print(f'\n    no finished target with ref time >= {a.long_sec}s')

    # ---- memory --------------------------------------------------------------
    rss_r = max(ref[t]['peak_memory_kb'] for t in common) / 1e6
    rss_n = max(new[t]['peak_memory_kb'] for t in common) / 1e6
    worst = max(common, key=lambda t: new[t]['peak_memory_kb'] / max(ref[t]['peak_memory_kb'], 1))
    print(f'\n[3] peak RSS  max: ref {rss_r:.2f} GB  new {rss_n:.2f} GB '
          f'({rss_n/rss_r:.3f}x)')
    print(f'    worst per-target ratio: {worst[:44]} '
          f'{new[worst]["peak_memory_kb"]/max(ref[worst]["peak_memory_kb"],1):.3f}x')
    return 0


if __name__ == '__main__':
    sys.exit(main())
