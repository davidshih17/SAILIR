#!/usr/bin/env python
"""Submit a PINNED paired timing test: both arms of each pair on ONE machine.

Why pinned rather than just simultaneous:
  Submitting two clusters at the same moment makes both arms see the same
  CLUSTER-wide load, but Condor still places them wherever slots are free, so
  they can land on machines with very different neighbours. On this cluster
  that matters a lot -- a single-core job sharing a box with a 28-core job runs
  ~1.5x slower with no CPU-allocation violation, purely from memory bandwidth
  and last-level cache pressure (observed on node43/node46, 2026-09-07).

  Pinning both arms of a pair to the same Machine makes the contention
  IDENTICAL and SYMMETRIC: they share every neighbour, including each other.
  Whatever is left in the A/B ratio is the code.

One machine per PAIR, different machines across pairs, so pairs do not pile
onto one box. Machines are chosen with the most free cores first, which leaves
slack in case something else claims capacity between selection and matching --
the real failure mode here is a pinned job sitting idle because its machine
filled up.

Timing needs only the LONG targets; correctness (bit-identity) is immune to
contention and should keep running over the full set, unpinned.

USAGE
  ab_submit_pinned.py --a-sub p3gval.sub --b-sub p3fval.sub \
      --targets long_targets.txt --tag 3d4 [--min-free 4] [--dry-run]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys


def free_cores():
    """(machine, free, busy) ranked QUIETEST first.

    Rank by BUSY cores ascending, not free descending. Contention comes from
    how many cores are WORKING on the box -- a single-core job shares memory
    bandwidth and last-level cache with every neighbour regardless of how many
    slots sit idle. Measured here: node43 has 12 free cores AND 28 busy ones,
    and a job there ran 1.58x; hexgpu1 has the most free cores (32) but 16
    busy. Meanwhile hexdl/node11/node13-15 are completely idle.
    Sorting by "most free" picks the loud machines. Sort by "least busy".
    """
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
        (free if f[1] == 'Unclaimed' else busy)[f[0]] = \
            (free if f[1] == 'Unclaimed' else busy).get(f[0], 0) + c
    machines = set(free) | set(busy)
    return sorted(((m, free.get(m, 0), busy.get(m, 0)) for m in machines),
                  key=lambda t: (t[2], -t[1]))


def write_sub(src, dest, expdir, target_file, machine):
    """Copy a submit file, repoint its EXPDIR/queue, and pin the Machine."""
    txt = open(src).read()
    old = re.search(r'EXPDIR=(\S+)', txt)
    if old:
        txt = txt.replace(old.group(1), expdir)
    txt = re.sub(r'^(log|output|error)\s*=.*$',
                 lambda m: m.group(0).replace(old.group(1), expdir) if old else m.group(0),
                 txt, flags=re.M)
    txt = re.sub(r'queue\s+tag\s+from\s+\S+', f'queue tag from {target_file}', txt)
    # the pin itself
    txt = re.sub(r'^requirements\s*=.*$', '', txt, flags=re.M | re.I)
    txt = txt.replace('queue tag from', f'requirements = (Machine == "{machine}")\nqueue tag from')
    open(dest, 'w').write(txt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a-sub', required=True)
    p.add_argument('--b-sub', required=True)
    p.add_argument('--targets', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--min-free', type=int, default=4,
                   help='only pin to machines with at least this many free cores')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--only', default=r'^node\d+\.',
                   help='regex a machine name must match to be pinnable. Default: '
                        'ordinary compute nodes only. GPU/special boxes (hexgpu1/2, '
                        'hexdl) advertise free cores and rank as QUIETEST, but do not '
                        'match ordinary jobs -- a pair pinned there sits IDLE forever.')
    a = p.parse_args()

    targets = [t.strip() for t in open(a.targets) if t.strip()]
    machines = [(m, f, b) for m, f, b in free_cores()
                if f >= a.min_free and re.search(a.only, m)]
    if len(machines) < len(targets):
        print(f'WARNING: {len(targets)} pairs but only {len(machines)} machines with '
              f'>= {a.min_free} free cores; pairs will share machines.')
    if not machines:
        sys.exit('ERROR: no machine has enough free cores to pin a pair')

    print(f'{len(targets)} pair(s); pinning each to its own machine:')
    for i, t in enumerate(targets):
        machine, cores, busy = machines[i % len(machines)]
        tdir = f'{a.tag}_pair{i}'
        os.makedirs(f'{tdir}/logs', exist_ok=True)
        tf = f'{tdir}/target.txt'
        open(tf, 'w').write(t + '\n')
        print(f'  {t[:34]:34s} -> {machine.split(".")[0]} '
              f'({cores} free, {busy} busy)')
        for arm, sub in (('A', a.a_sub), ('B', a.b_sub)):
            dest = f'{tdir}/{arm}.sub'
            expdir = f'{tdir}/{arm}'
            os.makedirs(f'{expdir}/logs', exist_ok=True)
            write_sub(sub, dest, expdir, tf, machine)
            if a.dry_run:
                continue
            r = subprocess.run(['condor_submit', dest], capture_output=True, text=True, timeout=120)
            print(f'      {arm}: {r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr.strip()}')
    if a.dry_run:
        print('\n(dry run -- nothing submitted)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
