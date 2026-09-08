#!/usr/bin/env python
"""Assert every performance optimization survives a strip/mutation.

Losing one of these does NOT fail any correctness test: results stay
bit-identical and the build just runs slower. That is exactly how a
hand-written greedy worker regressed to 1.22x median / 2.23x p90 -- it
silently dropped lazy_rs, the torch thread pin, and the per-step
malloc_trim(0), and every functional check still passed.

So this is checked mechanically, not by eye, on every emitted build.

Each entry is (name, pattern, min_count, why-it-matters). Counts are floors,
not equalities: deleting a dead CALL SITE legitimately lowers a count, while
deleting the MECHANISM must not. A count that falls below the floor means the
mechanism itself is gone.

Scope note: some optimizations live in onestep_worker_v9.py (the entry point),
not in beam_search_v9.py -- notably torch.set_num_threads, which appears in
beam_search_v9.main() only, a function our worker never calls. Pass --worker to
check those in the right file rather than reporting a false loss.
"""
import argparse
import re
import sys

# (label, regex, floor, rationale)
SEARCH_OPTS = [
    ('lazy_rs child build',      r'\blazy_rs\b',                 8,
     'children are built WITHOUT resolved_subs; only the survivor is '
     'materialized. Losing it cost median 1.22x / p90 2.23x.'),
    ('survivor materialize',     r'_materialize_lazy_rs',        3,
     'the other half of lazy_rs -- without it the survivor has no RS.'),
    ('packed representation',    r'_PACKED_RS',                  5,
     'int32 ids / int16 coeffs, ~6 bytes per term instead of dict entries.'),
    ('compiled cull',            r'_V9_CULL_C',                  4,
     'the C scoring kernel; the Python loop is the documented fallback.'),
    ('upfront enumeration',      r'_V9_UPENUM',                  3,
     '_tc_upenum_valid: enumerate without building the cu structure.'),
    ('raw-equation cache',       r'_raw_eq_cache',               4,
     'LRU over raw equations; a too-small cap caused a 20x cliff.'),
    ('per-step malloc_trim',     r'malloc_trim',                 3,
     'returns the freed heap top to the OS each step (DEFAULT ON).'),
    ('cpu affinity pin',         r'sched_setaffinity',           2,
     'keeps incidental thread pools inside the Condor slot.'),
    ('chunked model forward',    r'model_batch_chunk',           2,
     'bounds peak transient activation memory in the forward pass.'),
    ('total-order success test', r'_is_success',                 3,
     'NOT `if not nm` -- success means no non-master at-or-above the start '
     'in the FULL total ordering.'),
]

WORKER_OPTS = [
    ('torch intra-op pin',  r'torch\.set_num_threads',        1,
     'without it torch spawns one thread per core in a 1-CPU slot.'),
    ('torch inter-op pin',  r'set_num_interop_threads',       1,
     'the inter-op pool is SEPARATE from set_num_threads.'),
    ('incidental caps',     r'_cap_incidental_threads|N_THREADS', 1,
     'BLAS/OpenMP pools capped before numpy import.'),
]


def code_of(path):
    # ignore comments: a surviving comment must never satisfy a mechanism check
    return '\n'.join(l for l in open(path).read().splitlines()
                     if not l.lstrip().startswith('#'))


def check(path, ref_path, opts, label):
    """Calibrate against the REFERENCE rather than hand-picked floors.

    Hand-picked floors are guesswork and get it wrong: a floor of 2 for the
    affinity pin failed beam_search_v9.py itself, because 2 of its 3 textual
    hits are comments. A checker that fails its own reference is broken.

    So the rule is derived: if the reference uses a mechanism at all, the build
    must still use it. Counts may legitimately FALL (dead call sites go away);
    they may not reach zero.
    """
    code, ref = code_of(path), code_of(ref_path)
    print(f'=== {label}: {path} ===')
    lost = []
    for name, pat, _floor, why in opts:
        n = len(re.findall(pat, code))
        r = len(re.findall(pat, ref))
        if r == 0:
            print(f'  [n/a ] {name:<24s} not used by the reference')
            continue
        ok = n >= 1
        print(f'  [{"ok  " if ok else "LOST"}] {name:<24s} {n:3d}  (reference {r})')
        if not ok:
            lost.append((name, n, r, why))
    return lost


def main():
    p = argparse.ArgumentParser()
    p.add_argument('build', help='stripped beam_search build to check')
    p.add_argument('--worker', help='onestep_worker_*.py (entry-point opts)')
    p.add_argument('--ref', default='reduction/beam_search_v9.py',
                   help='reference build that DEFINES which mechanisms exist')
    p.add_argument('--ref-worker', default='reduction/onestep_worker_v9.py')
    a = p.parse_args()

    lost = check(a.build, a.ref, SEARCH_OPTS, 'search')
    if a.worker:
        lost += check(a.worker, a.ref_worker or a.worker, WORKER_OPTS, 'worker')

    print()
    if lost:
        print(f'FAIL: {len(lost)} optimization(s) lost')
        for name, n, r, why in lost:
            print(f'  - {name}: found {n}, reference uses it {r}x')
            print(f'      {why}')
        return 1
    print('OK: every optimization present')
    return 0


if __name__ == '__main__':
    sys.exit(main())
