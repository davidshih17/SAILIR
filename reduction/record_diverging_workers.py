"""Record workers whose reduction is DIVERGING, for later study.

Signature (measured on g1023, 2026-09-07): the max weight climbs away from the
start instead of descending, non-masters explode, and the substitution store
grows without bound. One worker reached mw=(201,190) with nm=2107 and rs_vsz
536,338 on a problem that starts at weight 12.

Read-only: this NEVER kills anything. Killing a worker externally would stall
the orchestrator, which keeps the integral in `pending` waiting for an output
file that will never appear -- the built-in straggler path is the only safe way
to kill, and it is disabled when --straggler-timeout exceeds 1e8.

Usage: record_diverging_workers.py <run_dir> [--nm-min N] [--out FILE]
"""
import argparse
import glob
import json
import os
import re
import sys

LINE = re.compile(r'\[v6 step +(\d+)\].*?best mw=\((\d+), (\d+), \(([^)]*)\)\).*?'
                  r'nm=(\d+).*?rs_vsz=(\d+).*?t_total=([0-9.]+)')
TAG = re.compile(r'async_\d+_(.+)$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--nm-min', type=int, default=500)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.run_dir, 'diverging_workers.jsonl')

    rows = []
    for f in glob.glob(os.path.join(a.run_dir, 'work/logs/*.out')):
        s = open(f, errors='ignore').read()
        done = 'SUCCESS in' in s
        ms = LINE.findall(s)
        if not ms:
            continue
        step, r, sN, absv, nm, rsv, t = ms[-1]
        if int(nm) < a.nm_min:
            continue
        first = ms[0]
        b = os.path.basename(f)[:-4]
        m = TAG.match(b)
        rows.append(dict(
            integral=m.group(1) if m else b,
            log=b, finished=done,
            nm=int(nm), nm_start=int(first[4]),
            max_w_r=int(r), max_w_s=int(sN), max_w_r_start=int(first[1]),
            step=int(step), rs_vsz=int(rsv), t_total=float(t),
            steps_logged=len(ms)))
    rows.sort(key=lambda x: -x['nm'])
    with open(out, 'w') as fh:
        for x in rows:
            fh.write(json.dumps(x) + '\n')

    print(f'{len(rows)} diverging workers (nm >= {a.nm_min}) -> {out}')
    if rows:
        print(f'{"integral":34s} {"nm":>6} {"start":>6} {"mw_r":>6} {"start":>6} '
              f'{"step":>6} {"rs_vsz":>9} {"t":>8}  fin')
        for x in rows[:12]:
            print(f'{x["integral"][:32]:34s} {x["nm"]:>6} {x["nm_start"]:>6} '
                  f'{x["max_w_r"]:>6} {x["max_w_r_start"]:>6} {x["step"]:>6} '
                  f'{x["rs_vsz"]:>9,} {x["t_total"]:>7.0f}s  {x["finished"]}')
        n_esc = sum(1 for x in rows if x['max_w_r'] > 2 * x['max_w_r_start'])
        print(f'\n  weight MORE THAN DOUBLED from its start: {n_esc}/{len(rows)}')
        print(f'  still running: {sum(1 for x in rows if not x["finished"])}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
