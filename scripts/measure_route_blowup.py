#!/usr/bin/env python
"""How many terms does a symmetry route produce, as a function of numerator degree s?

THE CONCERN. Routing rewrites integral I as a combination over its symmetry
image. Applying the map to a numerator of degree s is a MULTINOMIAL expansion
(canonicalize2.image_unsigned), so the term count can grow fast with s. If a
single high-s integral routes to hundreds or thousands of terms, routing is
EXPRESSION GROWTH, not reduction -- an IBP reduction of the same integral emits
a median of 6-8 terms, so routing would be strictly worse on that axis.

What matters is not the raw expansion but the FINAL rule size, after
cancellation and collection into the canonical representation. This measures
that, per integral, against s -- and separately reports how many emitted terms
are already masters or already in the cache, since those cost nothing
downstream.

Usage:
  measure_route_blowup.py --frontier <batches dir> [--per-bucket 4] [--timeout 120]
"""
import argparse
import glob
import os
import random
import signal
import sys
import time


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frontier', required=True,
                    help='directory of routing batch .txt files (real frontier)')
    ap.add_argument('--per-bucket', type=int, default=4)
    ap.add_argument('--timeout', type=int, default=120,
                    help='seconds per integral before giving up on it')
    ap.add_argument('--repo',
                    default='/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
    a = ap.parse_args()

    os.environ.setdefault('SAILIR_TOPOLOGY', 'gravity3L')
    os.environ['SAILIR_SECTOR_RANK'] = '1'
    sys.path.insert(0, a.repo)
    sys.path.insert(0, os.path.join(a.repo, 'reduction'))

    from sailir.topology import Topology
    from sailir import ibp_env as ie
    import topo_config as tc
    ie.init_from_topology(Topology.from_dir(tc.TOPO_DIR))
    cz = tc.canonicalize_module()
    ie.set_prime(cz.P)
    from canonical_masters import apply_canonical_masters
    apply_canonical_masters()
    from sailir.ibp_env import is_master
    from symmetry_route import canonical_monolithic_rule as route
    print(f'  solver={cz.__name__}  P={cz.P}  store={os.path.basename(tc.STORE_PKL)}')

    # collect frontier integrals, bucketed by s
    buckets = {}
    for f in sorted(glob.glob(os.path.join(a.frontier, '*.txt')))[:400]:
        for ln in open(f):
            ln = ln.strip()
            if not ln:
                continue
            I = tuple(int(x) for x in ln.split(','))
            s = -sum(x for x in I if x < 0)
            k = ('s=0-1' if s < 2 else 's=2-5' if s < 6 else 's=6-11'
                 if s < 12 else 's=12-19' if s < 20 else 's>=20')
            buckets.setdefault(k, []).append(I)

    signal.signal(signal.SIGALRM, _alarm)
    order = ['s=0-1', 's=2-5', 's=6-11', 's=12-19', 's>=20']
    print(f'\n  {"bucket":9s} {"s":>3s} {"terms":>6s} {"masters":>8s} {"secs":>7s}   integral')
    for k in order:
        pool = buckets.get(k, [])
        if not pool:
            continue
        for I in random.Random(0).sample(pool, min(a.per_bucket, len(pool))):
            s = -sum(x for x in I if x < 0)
            t0 = time.time()
            signal.alarm(a.timeout)
            try:
                r = route(I)
                dt = time.time() - t0
                signal.alarm(0)
            except _Timeout:
                print(f'  {k:9s} {s:3d} {"TIMEOUT":>6s} {"":>8s} {a.timeout:7d}   {I}')
                continue
            except Exception as e:                       # noqa: BLE001
                signal.alarm(0)
                print(f'  {k:9s} {s:3d} {"ERR":>6s}  {type(e).__name__}')
                continue
            if r is None:
                print(f'  {k:9s} {s:3d} {"survivor":>6s} {"":>8s} {dt:7.1f}   {I}')
            else:
                nm = sum(1 for J in r if is_master(J))
                print(f'  {k:9s} {s:3d} {len(r):6d} {nm:8d} {dt:7.1f}   {I}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
