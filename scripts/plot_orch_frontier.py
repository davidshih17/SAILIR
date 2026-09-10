#!/usr/bin/env python
"""Chart non-masters and the per-level frontier counts against iteration.

Reads the orchestrator log lines:

  [Iter 19] 0 masters, 44 non-masters | frontier=(L=10,r=12,s=0) x 5 | ...
             [maxw] L10: n=9 max=(r=12,s=0) L9: n=23 max=(r=11,s=0) ...

and plots, per iteration, the total non-master count and the `n=` counts for
each level in --levels.

Two logs, two numberings: a resumed run restarts [Iter 1], so plotting the raw
numbers would fold the second run back over the first. Pass logs in
chronological order and each one's iterations are offset past the previous
run's last iteration, with a dashed marker at the boundary. --separate keeps
them as distinct panels instead.

Usage:
  plot_orch_frontier.py --logs a.log b.log --out frontier.png
  plot_orch_frontier.py --logs a.log --levels 10 9 8 7 6 5 --separate
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TWO log formats. The orchestrator's [Iter N] prints gained a wall-clock stamp
# and an in-iteration offset -- "[15:23:45 Iter 42 +0.0s]" -- so a parser written
# for the bare "[Iter 42]" silently reads such a log as ZERO iterations and plots
# nothing, with no error. Accept both, and capture the stamp when present so
# iteration DURATION is available (it is unrecoverable from the old format).
ITER_RE = re.compile(
    r'\[(?:(\d\d:\d\d:\d\d) )?Iter (\d+)(?: \+[\d.]+s)?\]'
    r'\s+(\d+) masters,\s+(\d+) non-masters')
# "L9: n=23 max=(r=11,s=0)" -- the n= is the count of integrals at that level
MAXW_RE = re.compile(r'L(\d+):\s*n=(\d+)')


def parse(path, levels):
    """-> list of dicts, one per iteration that has BOTH an Iter and a maxw row.

    The maxw row follows its Iter line, so we hold the most recent Iter number
    and attach the next maxw row to it. An Iter line with no maxw row (the
    orchestrator does not print one every time) still contributes its
    non-master count; its level counts are left as None rather than 0, so a
    missing row is not drawn as a real drop to zero.
    """
    rows, cur = [], None
    for line in open(path, errors='ignore'):
        m = ITER_RE.search(line)
        if m:
            if cur is not None:
                rows.append(cur)
            cur = {'iter': int(m.group(2)), 'masters': int(m.group(3)),
                   'nm': int(m.group(4)), 'clock': m.group(1),
                   **{f'L{l}': None for l in levels}}
            continue
        if cur is not None and '[maxw]' in line:
            for lv, n in MAXW_RE.findall(line):
                key = f'L{lv}'
                if key in cur:
                    cur[key] = int(n)
    if cur is not None:
        rows.append(cur)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logs', nargs='+', required=True,
                   help='orchestrator logs, CHRONOLOGICAL order')
    p.add_argument('--levels', nargs='+', type=int,
                   default=[9, 8, 7, 6, 5])
    p.add_argument('--out', default=None)
    p.add_argument('--separate', action='store_true',
                   help='one panel per log instead of offsetting onto one axis')
    p.add_argument('--logy', action='store_true')
    a = p.parse_args()

    runs = []
    for f in a.logs:
        if not os.path.exists(f):
            sys.exit(f'missing log: {f}')
        r = parse(f, a.levels)
        if not r:
            print(f'  note: no [Iter] lines in {f}', file=sys.stderr)
        runs.append((os.path.basename(f), r))
        nstamp = sum(1 for x in r if x.get('clock'))
        print(f'  {os.path.basename(f)}: {len(r)} iterations, '
              f'nm {r[0]["nm"] if r else "-"} -> {r[-1]["nm"] if r else "-"}'
              f'{f", {nstamp} timestamped" if nstamp else ""}')
        if not r:
            print(f'    WARNING: parsed ZERO iterations from {f} -- if the file '
                  f'is non-empty the log format has changed again', file=sys.stderr)

    n_panels = len(runs) if a.separate else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 4.2 * n_panels),
                             squeeze=False)
    axes = axes[:, 0]

    colors = plt.cm.viridis([i / max(1, len(a.levels) - 1)
                             for i in range(len(a.levels))])

    if a.separate:
        for ax, (name, rows) in zip(axes, runs):
            draw(ax, rows, a.levels, colors, offset=0)
            ax.set_title(name)
    else:
        ax = axes[0]
        offset = 0
        for k, (name, rows) in enumerate(runs):
            if not rows:
                continue
            draw(ax, rows, a.levels, colors, offset=offset,
                 label_suffix='' if k == 0 else None)
            if k > 0:
                ax.axvline(offset, color='0.4', ls='--', lw=1)
                ax.text(offset, ax.get_ylim()[1], f' {name}', fontsize=7,
                        va='top', color='0.35')
            offset += rows[-1]['iter']
        ax.set_title('orchestrator frontier vs iteration '
                     f'({len(runs)} run{"s" if len(runs) > 1 else ""}, '
                     'dashed = resume boundary)')

    for ax in axes:
        ax.set_xlabel('iteration (cumulative across runs)')
        ax.set_ylabel('count')
        if a.logy:
            ax.set_yscale('log')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, ncol=2)

    out = a.out or 'orch_frontier.png'
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f'  wrote {out}')


def draw(ax, rows, levels, colors, offset=0, label_suffix=''):
    x = [r['iter'] + offset for r in rows]
    ax.plot(x, [r['nm'] for r in rows], color='crimson', lw=2,
            label='non-masters (total)' if label_suffix is not None else None)
    for c, lv in zip(colors, levels):
        key = f'L{lv}'
        # keep None as a gap: the orchestrator does not print a maxw row every
        # iteration, and drawing those as 0 invents a crash to zero
        xs = [xi for xi, r in zip(x, rows) if r.get(key) is not None]
        ys = [r[key] for r in rows if r.get(key) is not None]
        if ys:
            ax.plot(xs, ys, color=c, lw=1.3,
                    label=f'{key}' if label_suffix is not None else None)


if __name__ == '__main__':
    main()
