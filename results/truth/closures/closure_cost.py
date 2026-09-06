"""Is SAILIR_LOAD_CLOSURE worth it, or should the p=101 corpus just re-run the
existing p=1009 infrastructure unchanged?

The recorder's cost is worker_replay -- rebuilding the target's truth closure.
The closure campaign already paid that cost for all 12,584 targets and its
status files record the seconds per target. Summing them gives EXACTLY what the
load path would save, so the decision needs no estimate.

Two caveats that make the honest number LARGER than this sum:
  * the campaign was BATCHED -- one linear system shared across a whole group of
    targets. A per-target recorder run rebuilds that system every time, so the
    per-target cost is an underestimate of what re-running would take.
  * 350 groups blew a 19 GB ceiling building those systems. That memory is paid
    per worker, so a per-target campaign meets the same wall.
"""
import glob
import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
OUT = f'{D}/v4_p101_18k'


def main():
    secs = []
    for f in glob.glob(f'{OUT}/status_group_*.jsonl'):
        for line in open(f, errors='ignore'):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('ok') and r.get('sec') is not None:
                secs.append(r['sec'])
    secs.sort()
    n = len(secs)
    if not n:
        print('no timings found')
        return 1
    tot = sum(secs)
    print(f'targets with a recorded solve time : {n:,}')
    print(f'TOTAL closure CPU time             : {tot/3600:,.1f} CPU-h')
    print(f'  median {secs[n//2]:.2f}s   p90 {secs[int(.9*(n-1))]:,.1f}s   '
          f'max {secs[-1]:,.0f}s')
    for f in (.5, .9, .99):
        i = int(f * (n - 1))
        share = sum(secs[i:]) / tot
        print(f'  the slowest {100*(1-f):4.0f}% of targets carry '
              f'{100*share:5.1f}% of the total')
    over = [s for s in secs if s > 3600]
    print(f'  targets over 1 h: {len(over):,}  '
          f'({sum(over)/3600:,.1f} CPU-h, {100*sum(over)/tot:.0f}% of total)')

    # THE DECISIVE SPLIT. In each group the FIRST target pays to build the
    # shared linear system and the rest ride it nearly free (observed: 6179s
    # then 0.53s, 0.28s, 0.24s...). A per-target recorder run has no group to
    # amortise across, so it pays a system build EVERY time. Compare the two
    # regimes directly instead of assuming the batched total carries over.
    first, rest = [], []
    for f in glob.glob(f'{OUT}/status_group_*.jsonl'):
        rows = []
        for line in open(f, errors='ignore'):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('ok') and r.get('sec') is not None:
                rows.append(r['sec'])
        if rows:
            first.append(rows[0])
            rest.extend(rows[1:])
    if first:
        fm = sorted(first)
        print(f'\nBATCHING EFFECT')
        print(f'  groups that produced output : {len(first):,}')
        print(f'  FIRST target per group      : median {fm[len(fm)//2]:,.1f}s '
              f'  total {sum(first)/3600:,.1f} CPU-h '
              f'({100*sum(first)/tot:.0f}% of all closure time)')
        if rest:
            rm = sorted(rest)
            print(f'  every SUBSEQUENT target     : median '
                  f'{rm[len(rm)//2]:.2f}s   total {sum(rest)/3600:,.1f} CPU-h '
                  f'({100*sum(rest)/tot:.0f}%)')
            per = sum(first) / len(first)
            print(f'\n  batched (what we paid)              : '
                  f'{tot/3600:,.0f} CPU-h')
            print(f'  per-target estimate (no amortisation): '
                  f'{per*n/3600:,.0f} CPU-h'
                  f'   = {per*n/max(tot,1):.1f}x more')

    # what the WALK phase costs, for scale: the ftcorrupt campaign is the only
    # measured reference for the recorder's post-closure loop.
    prev = f'{D}/../corrupt_corpus/logs'
    if os.path.isdir(prev):
        import re
        RE = re.compile(r'total_time=([\d.]+)s')
        t = []
        for p in glob.glob(f'{prev}/*.out'):
            m = RE.search(open(p, errors='ignore').read())
            if m:
                t.append(float(m.group(1)))
        if t:
            print(f'\nftcorrupt reference: {len(t):,} targets, '
                  f'{sum(t)/3600:,.1f} CPU-h TOTAL recorder time '
                  f'(closure + walk), median {sorted(t)[len(t)//2]:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
