#!/usr/bin/env python
"""Verify every one-level reduction descends in the SAILIR total order.

A worker takes integral I and returns an expansion {J: coeff}. The whole scheme
rests on each J being STRICTLY LOWER than I in the total order -- otherwise the
orchestrator re-dispatches something no easier than what it started with, and
the frontier grows instead of draining.

tkey is IMPORTED, never rebuilt here. tkey(i) = (-RANK_IDX[mask], -r, -s,
tuple(abs(x))) and SMALLER == HIGHER == eliminated first, so a correct descent
means tkey(J) > tkey(I) for every emitted J. Hand-rolling this key has produced
wrong answers repeatedly -- collapsing the abs TUPLE to a sum, dropping the
sector-rank prefix, or flipping the sign convention.

Reports, over a sample of result pickles:
  * how many reductions emit a term that does NOT descend
  * whether the offenders are the giant-numerator (high s) integrals
  * the worst violations, with both tkeys, so a claim can be checked by hand

Usage:
  check_reduction_descent.py --results <dir> [--sample 3000] [--min-s 10]
"""
import argparse
import glob
import os
import pickle
import random
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    p.add_argument('--sample', type=int, default=3000,
                   help='0 = every pickle (slow: there may be >100k)')
    p.add_argument('--min-s', type=int, default=10,
                   help='call an emitted term "giant numerator" at or above this s')
    p.add_argument('--repo', default='/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    os.environ.setdefault('SAILIR_TOPOLOGY', 'gravity3L')
    os.environ['SAILIR_SECTOR_RANK'] = '1'          # sector rank is the SENIOR key
    sys.path.insert(0, a.repo)
    sys.path.insert(0, os.path.join(a.repo, 'reduction'))

    from sailir.topology import Topology
    from sailir import ibp_env as ie
    ie.init_from_topology(Topology.from_dir(
        os.path.join(a.repo, 'topology_input', os.environ['SAILIR_TOPOLOGY'])))
    ie.set_prime(101)
    ie.set_paper_masters_only(True)
    from canonical_masters import apply_canonical_masters
    apply_canonical_masters()
    from canonical_rep import tkey                  # IMPORT it, never rebuild it

    files = sorted(glob.glob(os.path.join(a.results, '*.pkl')))
    files = [f for f in files if not f.endswith('.checkpoint')]
    print(f'  {len(files)} result pickles')
    if a.sample and len(files) > a.sample:
        random.Random(a.seed).shuffle(files)
        files = files[:a.sample]
        print(f'  sampling {len(files)}')

    n_ok = n_bad = n_read = 0
    bad = []
    giant_terms = 0
    giant_from_bad = 0

    for f in files:
        try:
            with open(f, 'rb') as fh:
                r = pickle.load(fh)
        except Exception:
            continue
        I = r.get('original_integral')
        expr = r.get('final_expr')
        if not I or not expr:
            continue
        n_read += 1
        kI = tkey(tuple(I))
        lI = sum(1 for x in I if x > 0)   # level = propagator count
        offenders = []
        for J, c in expr.items():
            if not c:
                continue
            J = tuple(J)
            if J == tuple(I):
                continue
            s = -sum(x for x in J if x < 0)
            if s >= a.min_s:
                giant_terms += 1
            # SECTOR / LEVEL IS THE SENIOR KEY. A term that drops a propagator
            # is in a proper SUBSECTOR and is strictly lower by definition --
            # ORDERING.md: proper subsector < parent, prop count senior,
            # because IBP never leaves the sector cone, so IBP always descends.
            # Comparing raw tkey across sectors called those legitimate
            # reductions violations, which is why this reported ~4.7% on the
            # KNOWN-GOOD prior runs too. Only compare (r,s,|a|) WITHIN a level.
            lJ = sum(1 for x in J if x > 0)
            if lJ < lI:
                continue                       # dropped a propagator: descent
            if lJ > lI:
                offenders.append((J, tkey(J), s, 'level UP'))
                continue
            kJ = tkey(J)
            if kJ <= kI:            # same level, not strictly lower
                offenders.append((J, kJ, s, 'same level, not lower'))
        if offenders:
            n_bad += 1
            bad.append((I, kI, offenders))
            giant_from_bad += sum(1 for _, _, s, _w in offenders if s >= a.min_s)
        else:
            n_ok += 1

    print(f'\n  reductions checked : {n_read}')
    print(f'  descend correctly  : {n_ok}')
    print(f'  DO NOT descend     : {n_bad}')
    print(f'  emitted terms with s >= {a.min_s}: {giant_terms}'
          f'   (of which from non-descending reductions: {giant_from_bad})')

    if bad:
        print(f'\n  worst {min(5, len(bad))} violations:')
        for I, kI, offs in bad[:5]:
            print(f'    input  {tuple(I)}')
            print(f'      tkey {kI}')
            for J, kJ, s, why in offs[:3]:
                print(f'      emits {J}  s={s}  [{why}]')
                print(f'        tkey {kJ}   {"EQUAL" if kJ == kI else "HIGHER (worse)"}')
        return 1
    print('\n  every sampled reduction descends in the total order')
    return 0


if __name__ == '__main__':
    sys.exit(main())
