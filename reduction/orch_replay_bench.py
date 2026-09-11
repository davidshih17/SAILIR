#!/usr/bin/env python
"""ORCHESTRATOR REPLAY BENCHMARK (2026-07-30).

Replays a real arm's banked history through the ACTUAL orchestrator iteration
code (run_routing_cascade / compute_to_submit / apply_substitutions_delta —
the same module functions production calls) with Condor submission and
sleeping stubbed out. Worker results are delivered in their true arrival
order (file mtimes), batched into simulated poll windows.

Measures per-iteration phase times (deliver / fold / route / dispatch-set) and
writes a report; for a COMPLETED source arm the final expression is compared
against its banked reduction.pkl (hard correctness gate).

Usage:
  SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1 [SAILIR_DELTA_SUBS=1]
  orch_replay_bench.py --source results/gr_reduce/g1017 --target 'i1,...,iN'
      --bench-dir /tmp/bench_g1017 --window 5 [--max-iters N] [--gate]

SAILIR_ROUTE_CONDOR is forced OFF inside the bench (no cluster contact).
"""
import argparse
import glob
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))

os.environ['SAILIR_ROUTE_CONDOR'] = '0'          # never touch the cluster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--bench-dir', required=True)
    ap.add_argument('--window', type=float, default=5.0)
    ap.add_argument('--max-iters', type=int, default=10**9)
    ap.add_argument('--max-concurrent', type=int, default=1000)
    ap.add_argument('--gate', action='store_true',
                    help='compare final expr against source reduction.pkl')
    args = ap.parse_args()

    # --- init exactly as the orchestrator does ---
    from sailir import ibp_env
    from sailir.topology import Topology
    import topo_config as _tc
    ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
    ibp_env.set_prime(1009)
    ibp_env.set_paper_masters_only(True)
    from canonical_masters import apply_canonical_masters
    apply_canonical_masters()
    import hierarchical_reduction as HR
    if os.environ.get('SAILIR_PERSISTENT_POOL', '0') == '1':
        HR.init_route_pool()          # fork EARLY, before the big state loads

    class A:                                     # the args fields HR uses
        prime = 1009
        use_symmetry = True
        symmetry_staged = False
        max_concurrent = args.max_concurrent
    hargs = A()

    src = Path(args.source)
    bench = Path(args.bench_dir)
    (bench / 'results').mkdir(parents=True, exist_ok=True)

    # --- arrival stream: banked results in true arrival order ---
    t0 = time.time()
    files = glob.glob(str(src / 'work' / 'results' / '*.pkl'))
    stream = []
    for f in files:
        try:
            with open(f, 'rb') as fh:
                r = pickle.load(fh)
        except Exception:
            continue
        integ = r.get('original_integral')
        if integ is None:
            continue
        rule = (r.get('final_expr', {integ: 1}) if r.get('success')
                else {integ: 1})
        stream.append((os.path.getmtime(f), tuple(integ), rule, f))
    stream.sort(key=lambda x: x[0])
    print(f"[bench] arrival stream: {len(stream)} results "
          f"({time.time()-t0:.0f}s to load)", flush=True)

    sym_memo = {}
    smp_src = src / 'work' / 'sym_memo.pkl'
    if smp_src.exists():
        with open(smp_src, 'rb') as f:
            sym_memo = pickle.load(f)
        print(f"[bench] preloaded {len(sym_memo)} routing rules", flush=True)
    sym_memo_path = bench / 'sym_memo.pkl'

    target = tuple(int(x) for x in args.target.split(','))
    expr = {target: 1}
    cache = {}
    pending = {}
    delta_keys = set()
    ext_running, ext_ts = set(), time.time() + 10**9   # disable condor_q in bench
    n_routed = 0
    sm_saved = len(sym_memo)

    import gc
    gc.collect()
    gc.freeze()

    phases = {'deliver': [], 'fold': [], 'route': [], 'dispatch': []}
    iters = []
    si = 0
    sim_t = stream[0][0] if stream else 0
    iteration = 0
    while iteration < args.max_iters:
        iteration += 1
        it0 = time.time()
        # deliver this window's arrivals (harvest equivalent)
        t = time.time()
        sim_t += args.window
        # fast-forward across arrival gaps: an empty poll window costs the
        # real orchestrator only its sleep, so skipping ahead preserves the
        # workload while sparing the bench thousands of no-op iterations
        if si < len(stream) and stream[si][0] > sim_t:
            sim_t = stream[si][0]
        delivered = 0
        while si < len(stream) and stream[si][0] <= sim_t:
            _, integ, rule, f = stream[si]
            cache[integ] = rule
            delta_keys.add(integ)
            pending.pop(integ, None)
            dst = bench / 'results' / os.path.basename(f)
            if not dst.exists():
                try:
                    os.link(f, dst)
                except OSError:
                    pass
            si += 1
            delivered += 1
        phases['deliver'].append(time.time() - t)

        t = time.time()
        if os.environ.get('SAILIR_DELTA_SUBS', '0') == '1':
            expr, delta_keys, _ = HR.apply_substitutions_delta(
                expr, cache, delta_keys, 1009)
        else:
            expr = HR.apply_substitutions(expr, cache, 1009)
            delta_keys = set()
        phases['fold'].append(time.time() - t)

        t = time.time()
        expr, delta_keys, n_routed, sm_saved = HR.run_routing_cascade(
            expr, cache, sym_memo, pending, delta_keys, hargs, bench,
            iteration, n_routed, sym_memo_path, sm_saved)
        phases['route'].append(time.time() - t)

        t = time.time()
        to_submit, ext_running, ext_ts = HR.compute_to_submit(
            expr, cache, pending, delta_keys, bench, args.max_concurrent,
            ext_running, ext_ts)
        for i in to_submit:                       # stubbed submission
            pending[i] = (None, bench / 'x', 0, 1)
        phases['dispatch'].append(time.time() - t)

        iters.append(time.time() - it0)
        if iteration % 200 == 0:
            gc.collect()
            gc.freeze()
        if iteration % 100 == 0 or delivered > 500:
            nm = len(HR.get_non_masters(expr))
            print(f"[bench iter {iteration}] delivered={delivered} "
                  f"stream {si}/{len(stream)} nm={nm} "
                  f"pending={len(pending)} cache={len(cache)} "
                  f"it={iters[-1]:.2f}s", flush=True)
        if si >= len(stream) and not delta_keys:
            break

    def pct(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(len(v) * p))] if v else 0

    print("\n[bench] ===== REPORT =====")
    print(f"iterations: {len(iters)}  wall: {sum(iters):.0f}s")
    print(f"iter time  med={pct(iters,.5):.2f}s p90={pct(iters,.9):.2f}s "
          f"max={max(iters):.1f}s")
    for k, v in phases.items():
        print(f"  {k:9s} total={sum(v):7.1f}s med={pct(v,.5):.3f}s "
              f"p90={pct(v,.9):.3f}s max={max(v):.1f}s")
    nm_final = HR.get_non_masters(expr)
    print(f"final: {len(expr)} terms, {len(nm_final)} non-masters")

    if args.gate:
        ref = pickle.load(open(src / 'reduction.pkl', 'rb'))
        ea = {tuple(k): v % 1009 for k, v in ref['final_expr'].items()
              if v % 1009}
        eb = {k: v % 1009 for k, v in expr.items() if v % 1009}
        print(f"[gate] {'IDENTICAL' if ea == eb else 'MISMATCH'} "
              f"({len(ea)} vs {len(eb)} terms)")


if __name__ == '__main__':
    main()
