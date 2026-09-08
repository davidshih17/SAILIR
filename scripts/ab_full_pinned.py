#!/usr/bin/env python
"""Full paired A/B over EVERY target, each pair pinned to ONE machine.

Two submits total, not 2xN: Condor's multi-variable queue reads
`<target> <machine>` pairs from a file, and `requirements = (Machine == "$(mach)")`
pins each job. Arm A and arm B read the SAME file, so target i gets the same
machine in both arms -- that is what makes the pair comparable.

Why pinning is required, not merely nice:
  Submitting two clusters simultaneously is NOT enough. Measured 2026-09-07 over
  125 targets: only 2 of 116 pairs happened to land on the same host, and the
  A/B ratios ranged 0.64x-2.44x on builds whose results were bit-identical
  step for step. Host-to-host neighbour load dominates everything else.
  A single-core job sharing a box with a 28-core neighbour runs ~1.5x slower
  with no CPU-allocation violation (memory bandwidth + last-level cache).

Machine choice: LEAST BUSY first, not most free -- node43 had 12 free cores AND
28 busy ones. And only ordinary compute nodes: hexgpu1/2 and hexdl advertise
idle cores, so they rank as "quietest", but they do not match ordinary jobs and
a pair pinned there sits IDLE forever (observed: 4 of 6 pairs stuck).

Pairs are spread round-robin over eligible machines; with more pairs than
machines several pairs share a box, which is fine -- what matters is that the
two ARMS of a pair share it.
"""
import argparse
import os
import re
import subprocess
import sys


def machines(min_free, only):
    out = subprocess.run(['condor_status', '-af', 'Machine', 'State', 'Cpus'],
                         capture_output=True, text=True, timeout=120).stdout
    free, busy = {}, {}
    for line in out.splitlines():
        f = line.split()
        if len(f) < 3:
            continue
        try:
            c = int(f[2])
        except ValueError:
            continue
        d = free if f[1] == 'Unclaimed' else busy
        d[f[0]] = d.get(f[0], 0) + c
    cand = [(m, free.get(m, 0), busy.get(m, 0)) for m in set(free) | set(busy)
            if free.get(m, 0) >= min_free and re.search(only, m)]
    return sorted(cand, key=lambda t: (t[2], -t[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a-sub', required=True)
    p.add_argument('--b-sub', required=True)
    p.add_argument('--a-dir', required=True)
    p.add_argument('--b-dir', required=True)
    p.add_argument('--targets', required=True)
    p.add_argument('--min-free', type=int, default=2)
    p.add_argument('--only', default=r'^node\d+\.')
    p.add_argument('--pairs-file', default='ab_pairs.txt')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    tg = [t.strip() for t in open(a.targets) if t.strip()]
    ms = machines(a.min_free, a.only)
    if not ms:
        sys.exit('ERROR: no eligible machine')
    # 2 cores per pair; spread pairs so a machine is not oversubscribed by us
    slots = []
    for m, f, b in ms:
        slots += [m] * max(1, f // 2)
    with open(a.pairs_file, 'w') as fh:
        for i, t in enumerate(tg):
            fh.write(f'{t} {slots[i % len(slots)]}\n')
    print(f'{len(tg)} pairs over {len(set(slots))} machines '
          f'(least-busy first, {a.only!r} only)')
    for m, f, b in ms[:6]:
        print(f'    {m.split(".")[0]:9s} free={f:3d} busy={b:3d}')

    for arm, sub, out in (('A', a.a_sub, a.a_dir), ('B', a.b_sub, a.b_dir)):
        txt = open(sub).read()
        old = re.search(r'EXPDIR=(\S+)', txt).group(1)
        txt = txt.replace(old, out)
        txt = re.sub(r'^requirements\s*=.*$', '', txt, flags=re.M | re.I)
        txt = re.sub(r'queue\s+tag\s+from\s+\S+',
                     f'requirements = (Machine == "$(mach)")\n'
                     f'queue tag, mach from {a.pairs_file}', txt)
        dest = f'{out}.sub'
        open(dest, 'w').write(txt)
        os.makedirs(f'{out}/logs', exist_ok=True)
        if a.dry_run:
            print(f'  [{arm}] would submit {dest}')
            continue
        r = subprocess.run(['condor_submit', dest], capture_output=True, text=True, timeout=180)
        print(f'  [{arm}] {(r.stdout or r.stderr).strip().splitlines()[-1]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
