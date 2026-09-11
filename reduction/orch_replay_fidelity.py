#!/usr/bin/env python
"""FIDELITY REPLAY of the v22 orchestrator segment (2026-07-31).

Goal: reproduce v22's per-iteration wall times faithfully enough that
interventions can be trusted. Design (user-approved):

  1:1 ITERATIONS  — replay iteration i delivers exactly the result files
                    whose mtimes fall in production iteration i's window
                    (windows = consecutive t= stamps in orch_v22.log).
  LAUNCH STATE    — starting cache = the first N_launch results by mtime
                    (N_launch = the resume count printed in v22's log);
                    routing table = composed(sym_memo_raw_backup.pkl), the
                    closest surviving snapshot to v22's launch table
                    (approximation; stated in the report).
  REAL COSTS      — fold / routing scans / route solves (pool) / harvest
                    exists()+load / dedupe listdir run for real.
  CHARGED COSTS   — production's Condor routing waves: the replay computes
                    the solves locally (pool) to keep state exact, and the
                    report shows BOTH raw replay time and a charged time
                    where the local wave compute is replaced by the wave
                    durations v22's log actually recorded in that window.

Verification output: per-iteration CSV (T_prod, T_replay_raw, T_replay_chg,
phases, delivered) + rank correlation + side-by-side table of all production
iterations >300s. No intervention flags here — this IS the baseline.
"""
import os
import re
import sys
import glob
import time
import pickle
from datetime import datetime
from pathlib import Path

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))
os.environ['SAILIR_ROUTE_CONDOR'] = '0'    # big waves solved locally; charged
P = 1009


class _TSWriter:
    """Prefix every output line with a wall-clock timestamp (covers prints
    from the shared hierarchical_reduction functions too)."""
    def __init__(self, f):
        self._f = f
        self._nl = True

    def write(self, s):
        for part in s.splitlines(True):
            if self._nl:
                self._f.write(time.strftime("[%m-%d %H:%M:%S] "))
            self._f.write(part)
            self._nl = part.endswith("\n")

    def flush(self):
        self._f.flush()


sys.stdout = _TSWriter(sys.stdout)
sys.stderr = _TSWriter(sys.stderr)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench-dir', required=True)
    ap.add_argument('--max-iters', type=int, default=10**9)
    ap.add_argument('--checkpoint-iters', default='',
                    help='comma list of iterations AFTER which full replay '
                         'state is saved to results/orch_bench/ckpts/ '
                         '(state is node-independent; produce on any node)')
    ap.add_argument('--pace', choices=['none', 'real'], default='none',
                    help="'real': deliver each iteration's arrivals at the "
                         'same relative wall-clock offset as production '
                         '(sleeps between iterations like the live poll '
                         'loop) — the honest keep-up test for async mode')
    ap.add_argument('--from-ckpt', default=None,
                    help='start from a saved checkpoint: skips launch build, '
                         'the 3h launch wave and all earlier iterations; '
                         'workload (stamps/stream) is frozen in the ckpt so '
                         'every experiment replays the identical segment')
    args = ap.parse_args()

    _T0 = time.time()
    _SU = {}                       # startup phase accounting

    def _mark(name, t_start):
        _SU[name] = _SU.get(name, 0.0) + (time.time() - t_start)
        print(f"[fid] phase '{name}' {_SU[name]:.1f}s "
              f"(t+{time.time()-_T0:.0f}s)", flush=True)

    G = Path(ROOT) / "results/gr_reduce/g1023"
    bench = Path(args.bench_dir)
    (bench / 'results').mkdir(parents=True, exist_ok=True)
    ckpt_iters = {int(x) for x in args.checkpoint_iters.split(',') if x}
    ckpt_dir = Path(ROOT) / "results/orch_bench/ckpts"
    if ckpt_iters:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- launch state: resume from checkpoint, or build from scratch ----
    CK = None
    if args.from_ckpt:
        t0 = time.time()
        with open(args.from_ckpt, 'rb') as f:
            CK = pickle.load(f)
        print(f"[fid] checkpoint loaded in {time.time()-t0:.0f}s: resuming "
              f"after iter {CK['it']} (stream {CK['si']}, cache "
              f"{len(CK['cache'])}, table {len(CK['sym_memo'])})", flush=True)
        stamps, wave_events = CK['stamps'], CK['wave_events']
        launch_files, stream_files = CK['launch_files'], CK['stream_files']
    else:
        # production timeline from orch_v22.log
        _tp = time.time()
        PAT = re.compile(r"\[Iter (\d+)\] \d+ masters.*t=(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
        WAVE = re.compile(r"\[route-condor\] merged \d+ routings .* in (\d+)s")
        stamps = []          # (iter, epoch)
        wave_events = []     # (epoch_estimate, seconds)
        last_epoch = None
        for raw in open(G / "logs/orch_v22.log", errors="replace"):
            for ln in raw.split("\r"):
                m = PAT.search(ln)
                if m:
                    last_epoch = datetime.strptime(
                        m.group(2), "%Y-%m-%d %H:%M:%S").timestamp()
                    stamps.append((int(m.group(1)), last_epoch))
                w = WAVE.search(ln)
                if w and last_epoch is not None:
                    wave_events.append((last_epoch, float(w.group(1))))
        print(f"[fid] production stamps: {len(stamps)} iterations, "
              f"{len(wave_events)} recorded condor waves", flush=True)
        _mark('log-parse', _tp)

        # v22's launch cache size, from its resume line
        n_launch = None
        for raw in open(G / "logs/orch_v22.log", errors="replace"):
            m = re.search(r"\[RESUME\] Loaded (\d+) cache entries", raw)
            if m:
                n_launch = int(m.group(1))
                break
        assert n_launch, "no resume line found"
        print(f"[fid] launch cache: {n_launch} results", flush=True)

        # arrival stream
        t0 = time.time()
        entries = []
        for f in glob.glob(str(G / "work/results/*.pkl")):
            try:
                entries.append((os.path.getmtime(f), f))
            except OSError:
                pass
        entries.sort()
        launch_files = entries[:n_launch]
        stream_files = entries[n_launch:]
        print(f"[fid] {len(launch_files)} launch + {len(stream_files)} "
              f"stream files ({time.time()-t0:.0f}s)", flush=True)
        _mark('list-mtimes', t0)

    def load_rule(path):
        try:
            with open(path, 'rb') as fh:
                r = pickle.load(fh)
        except Exception:
            return None, None
        integ = r.get('original_integral')
        if integ is None:
            return None, None
        return tuple(integ), (r.get('final_expr', {tuple(integ): 1})
                              if r.get('success') else {tuple(integ): 1})

    # ---- init env exactly as production ----
    from sailir import ibp_env
    from sailir.topology import Topology
    import topo_config as _tc
    ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
    ibp_env.set_prime(P)
    ibp_env.set_paper_masters_only(True)
    from canonical_masters import apply_canonical_masters
    apply_canonical_masters()
    import hierarchical_reduction as HR

    class A:
        prime = P
        use_symmetry = True
        symmetry_staged = False
        max_concurrent = 1000
    hargs = A()

    if CK is not None:
        # restore full state; relink bench/results to match stream position
        cache, sym_memo = CK['cache'], CK['sym_memo']
        expr, pending = CK['expr'], CK['pending']
        delta_keys, si = CK['delta_keys'], CK['si']
        n_routed, sm_saved = CK['n_routed'], CK['sm_saved']
        t0 = time.time()
        n_link = 0
        for mt, f in list(launch_files) + list(stream_files[:si]):
            dst = bench / 'results' / os.path.basename(f)
            if not dst.exists():
                try:
                    os.link(f, dst)
                    n_link += 1
                except OSError:
                    pass
        print(f"[fid] relinked {n_link} result files "
              f"({time.time()-t0:.0f}s)", flush=True)
        print(f"[fid] STARTUP COMPLETE (from ckpt) in "
              f"{time.time()-_T0:.1f}s", flush=True)
        return run_loop(args, bench, ckpt_iters, ckpt_dir, stamps,
                        wave_events, stream_files, launch_files, load_rule,
                        cache, sym_memo, expr, pending, delta_keys, si,
                        n_routed, sm_saved, CK['idx'])

    # ---- launch state ----
    _tp = time.time()
    cache = {}
    for mt, f in launch_files:
        integ, rule = load_rule(f)
        if integ is not None:
            cache[integ] = rule
            dst = bench / 'results' / os.path.basename(f)
            if not dst.exists():
                try:
                    os.link(f, dst)
                except OSError:
                    pass
    print(f"[fid] launch cache loaded: {len(cache)}", flush=True)
    _mark('load-launch-pkls', _tp)

    # launch routing table ~ composed(raw backup)  [stated approximation]
    _tp = time.time()
    with open(G / "work/sym_memo_raw_backup.pkl", 'rb') as f:
        raw_tab = pickle.load(f)
    comp = {}
    for top, rule in raw_tab.items():
        if rule is None or top in comp:
            continue
        stack = [top]
        while stack:
            j = stack[-1]
            if j in comp:
                stack.pop()
                continue
            r = raw_tab.get(j)
            deps = [k for k in r if k not in comp and raw_tab.get(k) is not None]
            if deps:
                stack.extend(deps)
                continue
            out = {}
            for k, c in r.items():
                sub = comp.get(k) if raw_tab.get(k) is not None else None
                if sub is None:
                    out[k] = (out.get(k, 0) + c) % P
                else:
                    for l, cl in sub.items():
                        v = (out.get(l, 0) + c * cl) % P
                        if v:
                            out[l] = v
                        else:
                            out.pop(l, None)
            comp[j] = {k: v for k, v in out.items() if v}
            stack.pop()
    sym_memo = {k: (comp[k] if v is not None else None)
                for k, v in raw_tab.items()}
    print(f"[fid] launch routing table: {len(sym_memo)} entries (composed "
          f"from raw backup — approximation of v22 launch state)", flush=True)
    _mark('compose-table', _tp)
    _tot = time.time() - _T0
    _acc = sum(_SU.values())
    print(f"[fid] STARTUP COMPLETE in {_tot:.1f}s = "
          + " + ".join(f"{k} {v:.1f}s" for k, v in _SU.items())
          + f" + other {_tot-_acc:.1f}s (accounted {100*_acc/_tot:.1f}%)",
          flush=True)

    target = (1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0)
    return run_loop(args, bench, ckpt_iters, ckpt_dir, stamps, wave_events,
                    stream_files, launch_files, load_rule, cache, sym_memo,
                    {target: 1}, {}, set(), 0, 0, len(sym_memo), 0)


def run_loop(args, bench, ckpt_iters, ckpt_dir, stamps, wave_events,
             stream_files, launch_files, load_rule, cache, sym_memo,
             expr, pending, delta_keys, si, n_routed, sm_saved, idx0):
    import hierarchical_reduction as HR

    class A:
        prime = P
        use_symmetry = True
        symmetry_staged = False
        max_concurrent = 1000
    hargs = A()

    ext_running, ext_ts = set(), time.time() + 10**9
    sym_memo_path = bench / 'sym_memo.pkl'

    import gc
    gc.collect()
    gc.freeze()

    # ---- replay production iterations 1:1 ----
    rows = []
    prev_epoch = stamps[idx0][1]
    csv = open(bench / "fidelity.csv", "w")
    csv.write("iter,T_prod,T_raw,T_charged,deliver,fold,route,dispatch,"
              "harvest,delivered,wave_charge\n")
    _pace0 = time.time()
    _epoch0 = stamps[idx0][1]
    for idx, (it, epoch) in enumerate(stamps):
        if idx <= idx0:
            continue
        if idx - idx0 > args.max_iters:
            break
        T_prod = epoch - prev_epoch
        if args.pace == 'real':
            _tgt = _pace0 + (epoch - _epoch0)
            _wait = _tgt - time.time()
            if _wait > 0:
                if _wait > 30:
                    print(f"[fid] pacing: sleeping {_wait:.0f}s to match "
                          f"production iter {it} window", flush=True)
                time.sleep(_wait)
            elif _wait < -60:
                print(f"[fid] pacing: RUNNING {-_wait:.0f}s BEHIND "
                      f"production at iter {it}", flush=True)
        it0 = time.time()
        # deliver: link files in this window (arrival), harvest below reads
        t = time.time()
        arrivals = []
        while si < len(stream_files) and stream_files[si][0] <= epoch:
            mt, f = stream_files[si]
            dst = bench / 'results' / os.path.basename(f)
            if not dst.exists():
                try:
                    os.link(f, dst)
                except OSError:
                    pass
            arrivals.append(dst)
            si += 1
        t_deliver = time.time() - t
        # harvest: REAL exists() + load for every arrival (production checks
        # its pending table; volume equivalent)
        t = time.time()
        for dst in arrivals:
            if not dst.exists():
                continue
            integ, rule = load_rule(dst)
            if integ is not None:
                cache[integ] = rule
                delta_keys.add(integ)
                pending.pop(integ, None)
        t_harvest = time.time() - t
        # fold
        t = time.time()
        expr, delta_keys, _ = HR.apply_substitutions_delta(
            expr, cache, delta_keys, P)
        t_fold = time.time() - t
        # routing cascade (pool for big waves; wave charge separately)
        t = time.time()
        expr, delta_keys, n_routed, sm_saved = HR.run_routing_cascade(
            expr, cache, sym_memo, pending, delta_keys, hargs, bench,
            it, n_routed, sym_memo_path, sm_saved)
        t_route = time.time() - t
        # dispatch set (real listdir dedupe) + stub submission
        t = time.time()
        to_submit, ext_running, ext_ts = HR.compute_to_submit(
            expr, cache, pending, delta_keys, bench, 1000,
            ext_running, ext_ts)
        for i in to_submit:
            pending[i] = (None, bench / 'x', 0, 1)
        t_dispatch = time.time() - t

        T_raw = time.time() - it0
        # charge: production condor-wave seconds recorded in this window
        wave_charge = sum(w for e, w in wave_events if prev_epoch < e <= epoch)
        T_charged = T_raw + wave_charge
        rows.append((it, T_prod, T_raw, T_charged))
        csv.write(f"{it},{T_prod:.1f},{T_raw:.2f},{T_charged:.2f},"
                  f"{t_deliver:.2f},{t_fold:.2f},{t_route:.2f},"
                  f"{t_dispatch:.2f},{t_harvest:.2f},{len(arrivals)},"
                  f"{wave_charge:.0f}\n")
        csv.flush()
        if T_prod > 120 or T_raw > 30 or it % 25 == 0:
            print(f"[fid iter {it}] prod={T_prod:.0f}s raw={T_raw:.1f}s "
                  f"charged={T_charged:.1f}s (fold={t_fold:.1f} "
                  f"route={t_route:.1f} disp={t_dispatch:.1f} "
                  f"harv={t_harvest:.1f} arr={len(arrivals)} "
                  f"wave+{wave_charge:.0f} "
                  f"inflight={len(HR._ASYNC_PENDING)})", flush=True)
        prev_epoch = epoch
        if it in ckpt_iters:
            t0c = time.time()
            ck = dict(idx=idx, it=it, si=si, expr=expr, cache=cache,
                      sym_memo=sym_memo, pending=pending,
                      delta_keys=delta_keys, n_routed=n_routed,
                      sm_saved=sm_saved, stamps=stamps,
                      wave_events=wave_events, launch_files=launch_files,
                      stream_files=stream_files)
            pth = ckpt_dir / f"ckpt_iter{it}.pkl"
            with open(str(pth) + '.tmp', 'wb') as fh:
                pickle.dump(ck, fh, protocol=4)
            os.replace(str(pth) + '.tmp', pth)
            print(f"[fid] checkpoint after iter {it} -> {pth} "
                  f"({time.time()-t0c:.0f}s)", flush=True)
        if it % 200 == 0:
            gc.collect()
            gc.freeze()

    csv.close()
    if os.environ.get('SAILIR_ASYNC_ROUTE', '0') == '1':
        # converge: collect in-flight solves, then one synchronous cascade
        t0d = time.time()
        os.environ['SAILIR_ASYNC_ROUTE'] = '0'
        HR.drain_async_routes(sym_memo)
        expr, delta_keys, n_routed, sm_saved = HR.run_routing_cascade(
            expr, cache, sym_memo, pending, delta_keys, hargs, bench,
            999999, n_routed, sym_memo_path, sm_saved)
        print(f"[fid] final async drain + sync cascade: "
              f"{time.time()-t0d:.0f}s", flush=True)
    with open(bench / "final_state.pkl", 'wb') as fh:
        pickle.dump({'expr': expr, 'n_cache': len(cache),
                     'n_sym_memo': len(sym_memo)}, fh)
    print(f"[fid] final state saved: {len(expr)} terms", flush=True)
    # ---- verification report ----
    import math
    prod = [r[1] for r in rows]
    chg = [r[3] for r in rows]

    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for j, i in enumerate(s):
            r[i] = j
        return r

    rp, rc = rank(prod), rank(chg)
    n = len(rows)
    if n > 2:
        d2 = sum((a - b) ** 2 for a, b in zip(rp, rc))
        rho = 1 - 6 * d2 / (n * (n * n - 1))
    else:
        rho = float('nan')
    print(f"\n[fid] ===== VERIFICATION =====")
    print(f"iterations compared: {n}")
    print(f"Spearman rank correlation (prod vs charged): {rho:.3f}")
    for name, v in (("prod", prod), ("charged", chg)):
        sv = sorted(v)
        print(f"  {name:8s} med={sv[n//2]:.1f}s p90={sv[int(n*.9)]:.1f}s "
              f"max={sv[-1]:.1f}s")
    print(f"\nproduction iterations >300s, side by side:")
    print(f"{'iter':>6} {'T_prod':>8} {'T_raw':>8} {'T_charged':>9}")
    for it, tp, tr, tc in rows:
        if tp > 300:
            print(f"{it:>6} {tp:>8.0f} {tr:>8.1f} {tc:>9.1f}")


if __name__ == '__main__':
    main()
