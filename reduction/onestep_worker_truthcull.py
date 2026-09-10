#!/usr/bin/env python3
"""onestep_worker_v8.py
=================================================================
Hierarchical-reduction worker that runs `beam_search_truthcull.beam_search_v5`
IN-PROCESS — exactly the way onestep_worker_v6.py runs v6, and exactly the way
the verified probes ran `python beam_search_truthcull.py --n-threads 8 --n-workers 8`.

WHY IN-PROCESS (and not a subprocess wrapper):
  A single process means beam_search_truthcull's import-time _cap_incidental_threads()
  runs in THIS process and confines it + its 8 forked enumerate workers to 8
  cores — byte-for-byte the CPU behavior the probes verified (8/8, zero holds).
  There is no pickle round-trip (so no `ModuleNotFoundError: sailir`), no stdout
  buffering (so the .out shows live progress and survives a kill), and the
  orchestrator-format result.pkl is written directly (so reaping just works).

v7 SETTINGS reproduced (== the success-only probe recipe):
  ENV (set BELOW, before importing beam_search_truthcull, because _SUCCESS_TOTAL /
       _BEAM_TOTAL / _ACTION_SELECT are read at module import):
       SAILIR_SUCCESS_TOTAL=1  SAILIR_ACTION_SELECT=maxweight
       SAILIR_STRIP_RAWS=1  SAILIR_PACKED_RS=1  SAILIR_END_OF_STEP_TRIM=1
       SAILIR_TABU_CAP=0  MALLOC_MMAP_THRESHOLD_=67108864  PYTHONUNBUFFERED=1
       (SAILIR_BEAM_TOTAL deliberately UNSET -> (r,s) beam + (r,s) maxweight)
  ARGV injection: --n-threads 8 --n-workers 8 prepended so the import-time
       _cap_incidental_threads() (which peeks sys.argv) applies the 8/8 thread
       caps + MKL GNU layer + 8-core affinity pin, identical to the probe.
  beam_search_v5 call: tabu=True, use_exprkeyed=False, iraws_keep_first=50,
       lazy_rs=True, max_actions=900, beam_sort='weight', model_batch_chunk=8,
       n_workers=8 (the fork pool).

CLI mirrors onestep_worker_v6.py so hierarchical_reduction.py dispatches to it
unchanged. Output keys match the orchestrator contract:
  success, original_integral, final_expr, path, restart_offsets, steps, time,
  prime, peak_memory_kb.
"""
import os
import sys

# ── v7 toggles: MUST be set before importing beam_search_truthcull (module-level
#    _SUCCESS_TOTAL/_BEAM_TOTAL/_ACTION_SELECT read os.environ at import). ──
os.environ.setdefault('PYTHONUNBUFFERED', '1')
os.environ.setdefault('MALLOC_MMAP_THRESHOLD_', '67108864')
os.environ['SAILIR_END_OF_STEP_TRIM'] = '1'
os.environ['SAILIR_TABU_CAP'] = '0'
os.environ['SAILIR_STRIP_RAWS'] = '1'
os.environ['SAILIR_PACKED_RS'] = '1'
os.environ['SAILIR_SUCCESS_TOTAL'] = '1'
os.environ.setdefault('SAILIR_ACTION_SELECT', 'maxweight')
# SAILIR_BEAM_TOTAL intentionally absent -> (r,s) beam + (r,s) maxweight.

# ── argv injection so beam_search_truthcull's import-time _cap_incidental_threads()
#    sees the production 8/8 config and applies thread-caps + 8-core pin. It
#    peeks sys.argv for --n-threads/--n-workers; our own parser uses
#    parse_known_args() below so these extra flags are ignored. ──
# CPU count from --v7-cpus (default 8). Peeked from sys.argv BEFORE importing
# beam_search_truthcull, because its import-time _cap_incidental_threads() reads the
# injected --n-threads/--n-workers to size the affinity pin + thread caps.
def _peek_v7_cpus(default=8):
    for i, a in enumerate(sys.argv):
        if a == '--v7-cpus' and i + 1 < len(sys.argv):
            try:
                return max(1, int(sys.argv[i + 1]))
            except ValueError:
                return default
        if a.startswith('--v7-cpus='):
            try:
                return max(1, int(a.split('=', 1)[1]))
            except ValueError:
                return default
    return default


N_THREADS = N_WORKERS = _peek_v7_cpus()
# Cap BLAS/OpenMP pools to the slot size BEFORE any numpy import (the
# symmetry-first block below imports sailir/numpy ahead of beam_search_truthcull's
# thread caps; an uncapped OpenBLAS pool would exceed RequestCpus -> hold).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, str(N_THREADS))
if '--n-threads' not in sys.argv:
    sys.argv += ['--n-threads', str(N_THREADS)]
if '--n-workers' not in sys.argv:
    sys.argv += ['--n-workers', str(N_WORKERS)]

import argparse
import pickle
import resource
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # .../SAILIR_phase2 (repo root) for `sailir`
sys.path.insert(0, str(_HERE))          # .../reduction for sibling modules

# ── SYMMETRY-FIRST (2026-07-31): routing on the same footing as SAILIR. ──
# Before paying the beam_search_truthcull/torch import + model load, try the symmetry
# route solve on the target. A non-None rule IS a valid one-step reduction
# (strictly descending in the total order by construction) and becomes this
# worker's result in seconds. None -> fall through to the full SAILIR path
# (the attempt cost is negligible vs beam search). Guarded by
# SAILIR_SYM_FIRST=1, which the orchestrator propagates into worker jobs.
if os.environ.get('SAILIR_SYM_FIRST', '0') == '1':
    def _sym_first():
        def peek(name):
            for i, a in enumerate(sys.argv):
                if a == name and i + 1 < len(sys.argv):
                    return sys.argv[i + 1]
                if a.startswith(name + '='):
                    return a.split('=', 1)[1]
            return None
        topo = peek('--topology')
        integ = peek('--integral')
        out = peek('--output')
        if not (topo and integ and out):
            return
        prime = int(peek('--prime') or 1009)
        t0 = time.time()
        from sailir.topology import Topology
        from sailir import ibp_env as _ie
        _ie.init_from_topology(Topology.from_dir(topo))
        _ie.set_prime(prime)
        _ie.set_paper_masters_only(True)
        if os.environ.get('SAILIR_SECTOR_RANK', '0') == '1':
            from canonical_masters import apply_canonical_masters
            apply_canonical_masters()
        from symmetry_route import canonical_monolithic_rule, tkey
        I = tuple(int(x) for x in integ.strip("'").strip('"').split(','))
        # TIME-LIMIT THE ROUTE. Routing cost is bimodal to an extreme degree:
        # measured over 104,383 integrals the MEDIAN routes in 1.2s and p90 in
        # 3.2min, but a tail runs for HOURS (one observed still going at 6.7h).
        # Unbounded here, a single such integral holds this worker for hours
        # before it even starts the beam search it was dispatched to do.
        # Timing out is lossless: we simply fall through to that beam search,
        # exactly as for any integral symmetry cannot route.
        _tl = int(os.environ.get('SAILIR_ROUTE_TIME_LIMIT', '300'))
        # NUMERATOR-DEGREE GATE. Checked BEFORE routing, so a skipped integral
        # costs nothing -- unlike the time limit, which only bails after burning
        # its full budget. Routing cost is bimodal and high-s is where the
        # pathological tail lives, so without this every high-s worker pays up
        # to the time limit before falling through. s is a PROXY, not the cause
        # (measured: s=24 -> 1 term, s=16 -> 6,435, s=20 -> 1,315,600); what it
        # buys is predictability, and everything sampled at s<=5 gave small
        # rules. This gate used to live in the orchestrator, which pre-screened
        # candidates before bulk routing; that block is gone, so it belongs here.
        _ms = int(os.environ.get('SAILIR_ROUTE_MAX_S', '5'))
        _s = -sum(x for x in I if x < 0)
        if _ms >= 0 and _s > _ms:
            print(f'[sym-first] s={_s} > {_ms} — skipping route, straight to '
                  f'beam search', flush=True)
            return
        # START MARKER. Every other sym-first line is printed AFTER the route
        # returns, so without this a worker stuck in the route logs NOTHING and
        # the silence is indistinguishable from a slow import, model load or
        # beam-search start. With it, the rule is simple: this line with no
        # following [sym-first] line means the route is still running -- and if
        # the gap exceeds the limit, the limit is NOT working.
        print(f'[sym-first] routing {",".join(str(x) for x in I)} '
              f'(s={_s}, limit {_tl}s) ...', flush=True)
        # ARM THE ALARM ONLY IMMEDIATELY BEFORE THE CALL IT BOUNDS, AND ALWAYS
        # DISARM IT IN A finally.
        #
        # This previously armed the alarm above the numerator-degree gate, and
        # that gate `return`s -- so every s>MAX_S worker walked out of sym-first
        # with a LIVE 300s alarm, started the beam search, and was interrupted
        # mid-search by SIGALRM raising TimeoutError from arbitrary code
        # (observed inside np.take_along_axis in _tc_upenum_candidates). The
        # worker died exit=1, wrote no result pkl, and its `pending` entry was
        # therefore never collected or cleared -- so its concurrency slot was
        # never returned. 610,012 frontier integrals are s>5, so this drained
        # the orchestrator to available_slots = max_concurrent - 9,997 = 3 and
        # the cluster fell to 26 running jobs.
        #
        # The handler is also RESTORED, not left installed: leaving it in place
        # means any later SIGALRM in this process still raises TimeoutError out
        # of unrelated code.
        import signal
        _prev_handler = None
        if _tl > 0:
            def _sym_alarm(signum, frame):
                raise TimeoutError('sym-first route exceeded the time limit')
            _prev_handler = signal.signal(signal.SIGALRM, _sym_alarm)
            signal.alarm(_tl)
        try:
            rule = canonical_monolithic_rule(I)
        except TimeoutError:
            print(f'[sym-first] route exceeded {_tl}s — proceeding to beam '
                  f'search', flush=True)
            return
        finally:
            # unconditional: cancel any pending alarm and put the previous
            # handler back, on EVERY exit path including the returns above.
            signal.alarm(0)
            if _prev_handler is not None:
                signal.signal(signal.SIGALRM, _prev_handler)
        el = time.time() - t0
        if rule is None:
            print(f'[sym-first] survivor after {el:.1f}s — proceeding to '
                  f'beam search', flush=True)
            return
        # OUTPUT SIZE CAP. A rule with more terms than this is expression
        # GROWTH, not reduction -- an IBP reduction of the same integral emits a
        # median of 6-8 terms, so the beam search is strictly better. Measured:
        # one route produced 1,315,600 terms against a 714,538-term expression.
        # Also previously enforced by the removed orchestrator block.
        _mt = int(os.environ.get('SAILIR_ROUTE_MAX_TERMS', '200'))
        if len(rule) > _mt:
            print(f'[sym-first] rule has {len(rule)} terms > {_mt} — expression '
                  f'growth, proceeding to beam search', flush=True)
            return
        k0 = tkey(I)
        # DESCENT IS MODULO THE TERMINAL SET, matching symmetry_rule. A term
        # that is a master or corner needs no further reduction wherever it sits
        # in the order, so it cannot close a worker/symmetry cycle. A raw tkey
        # assertion here would KILL THE WORKER on a rule the router legitimately
        # returns. ibp_env is initialised just above, so is_master is the real
        # terminal set (basis UNION corners in uncovered sectors).
        assert all(tkey(k) > k0 or _ie.is_master(k) for k in rule), \
            'sym-first rule neither descending nor terminal'
        result = {
            'success': True,
            'original_integral': I,
            'final_expr': {k: v % prime for k, v in rule.items()},
            'path': [],
            'restart_offsets': [],
            'steps': 0,
            'time': el,
            'prime': prime,
            'peak_memory_kb':
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'method': 'symmetry',
        }
        op = Path(out)
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(str(op) + '.tmp', 'wb') as f:
            pickle.dump(result, f)
        os.replace(str(op) + '.tmp', op)
        print(f'[sym-first] SYMMETRY-ROUTED in {el:.1f}s: {len(rule)} '
              f'terms -> {out}', flush=True)
        sys.exit(0)

    try:
        _sym_first()
    except SystemExit:
        raise
    except Exception as _e:
        print(f'[sym-first] error ({_e}) — falling through to beam search',
              flush=True)

# Import beam_search_truthcull FIRST. Its module top runs _cap_incidental_threads()
# (thread caps + 8-core affinity pin) BEFORE it (internally) imports numpy/torch,
# so torch's threadpool is created already-pinned to 8 cores.
#
# CRITICAL ORDER: os.sched_setaffinity(0, mask) pins only the CALLING thread and
# any threads created AFTERWARD. If torch (or sailir.classifier, which imports
# torch) is imported BEFORE this, torch spins up its threadpool across ALL cores,
# and the later main-thread-only pin canNOT reclaim those threads -> total CPU
# exceeds RequestCpus -> Condor holds the job. So beam_search_truthcull (hence its
# torch) must load before any other torch-importing module. This mirrors the
# standalone beam_search_truthcull.py order (_cap_incidental_threads() then import torch).
import beam_search_truthcull as bs7
from beam_search_truthcull import (beam_search_v5, replay_full_expr, _is_success,
                            _target_key, max_w12)
import torch
from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (set_prime, set_paper_masters_only, IBPEnvironment,
                            weight)
from sailir.classifier import IBPActionClassifier
from beam_search_utils import get_sector_mask


def main():
    p = argparse.ArgumentParser(
        description='in-process v7 onestep worker for hierarchical_reduction.py')
    p.add_argument('--topology', required=True)
    p.add_argument('--integral', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model-checkpoint', required=True)
    p.add_argument('--beam_width', type=int, default=40)
    p.add_argument('--max_steps', type=int, default=10**6)
    p.add_argument('--prime', type=int, default=1009)
    p.add_argument('--device', default='cpu')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--paper-masters-only',
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume-from', default=None)
    # Accepted for CLI parity with onestep_worker_v6 (some ignored — v7 fixes them).
    p.add_argument('--n_workers', type=int, default=N_WORKERS)
    p.add_argument('--random-target', action='store_true')
    p.add_argument('--dedup-beam-by-content',
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--beam-sort', default='weight')
    p.add_argument('--no-tabu', dest='tabu', action='store_false', default=True)
    p.add_argument('--no-lazy-rs', dest='lazy_rs', action='store_false', default=True)
    p.add_argument('--iraws-keep-first', type=int, default=50)
    p.add_argument('--no-exprkeyed', dest='use_exprkeyed',
                   action='store_false', default=False)
    p.add_argument('--checkpoint-path', default=None)
    p.add_argument('--no-checkpoint', action='store_true')
    p.add_argument('--checkpoint-interval', type=int, default=100)
    p.add_argument('--checkpoint-time-seconds', type=int, default=300)
    # parse_known_args: tolerate the injected --n-threads/--n-workers flags.
    args, _unknown = p.parse_known_args()

    # ── replicate beam_search_truthcull.main()'s setup VERBATIM ──────────────────
    torch.set_num_threads(N_THREADS)
    if (os.environ.get('SAILIR_CAP_INTEROP', '1') != '0'
            and (N_THREADS > 1 or N_WORKERS > 1)):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    t0 = time.time()
    topology = Topology.from_dir(args.topology)
    ibp_env.init_from_topology(topology)
    assert topology.n_denominators == bs7._TC_N_DEN, (
        f"topology has {topology.n_denominators} denominators but "
        f"SAILIR_TOPOLOGY={os.environ.get('SAILIR_TOPOLOGY', 'pentagonbox')!r} "
        f"configures {bs7._TC_N_DEN} — set SAILIR_TOPOLOGY to match --topology")
    set_prime(args.prime)
    set_paper_masters_only(args.paper_masters_only)
    if os.environ.get('SAILIR_SECTOR_RANK', '0') == '1':
        # Canonical-sector pipeline package: masters = symmetry-images of the paper
        # masters in OUR canonical sectors (canonical_masters.py). Without this the
        # merged corner of a Kira-mismatched orbit is unreducible and the worker
        # hangs (the m1/m3 stuck-corner pathology).
        from canonical_masters import apply_canonical_masters
        apply_canonical_masters()
    env = IBPEnvironment()

    # v7 module globals that beam_search_truthcull.main() sets (replicate exactly).
    bs7._V7_REGISTRY = bs7.IntegralRegistry()
    bs7._V7_PACKED_RS_CACHE = {}
    bs7._PACKED_RS = (os.environ.get('SAILIR_PACKED_RS', '0') == '1')

    # Variant-aware model construction (checkpoints/<name>/README.md): the
    # canon10x retrain is the `nosubs` variant (IBPActionClassifierNoSubs) —
    # same forward call-site (sub_* args accepted and ignored), but it MUST be
    # built with the training prime (1009): the coefficient encoder folds
    # coeffs > p/2 to negatives via self.prime, and the constructor default
    # (2^31-1) would silently break that. Older checkpoints (model_variant
    # None) keep the exact legacy construction.
    # NOTHING RANKS under truthminnew (it yields ONE action) or any model-free
    # score, so no checkpoint is needed. TRUTHCULL COPY ONLY -- production
    # onestep_worker_v8.py still loads unconditionally.
    if (os.environ.get('SAILIR_ACTION_SELECT') == 'truthminnew'
            or os.environ.get('SAILIR_ACTION_SCORE', 'model') != 'model'):
        print('[worker] model-free run -- no checkpoint loaded', flush=True)
        model = None
    else:
        ck = torch.load(args.model_checkpoint, map_location='cpu', weights_only=False)
        _variant = (ck.get('args') or {}).get('model_variant')
        if _variant == 'nosubs':
            from sailir.classifier_nosubs import IBPActionClassifierNoSubs
            model = IBPActionClassifierNoSubs(
                prime=(ck.get('args') or {}).get('prime', args.prime),
                n_indices=topology.n_indices,
                n_denominators=topology.n_denominators,
                n_ibp_ops=topology.n_actions,
            )
        else:
            # BUG FIX 2026-07-15: prime was omitted here since the 2026-05-11 public
            # release, silently constructing with the old default 2^31-1 while the
            # model was TRAINED at prime=1009 -- the coefficient sign-fold (residues
            # > p/2 -> negatives) then never fired at inference, flipping 8.2% of
            # top-1 decisions (analysis/probe_subs_sensitivity.py). `prime` is now a
            # REQUIRED keyword in every classifier constructor: omission crashes at
            # construction instead of degrading silently.
            model = IBPActionClassifier(
                prime=(ck.get('args') or {}).get('prime', args.prime),
                n_indices=topology.n_indices,
                n_denominators=topology.n_denominators,
                n_ibp_ops=topology.n_actions,
            )
        model.load_state_dict(ck['model_state_dict'])
        model.eval()

    integral_str = args.integral.strip("'").strip('"')
    start_int = tuple(int(x) for x in integral_str.split(','))
    start_w = weight(start_int)
    start_w12 = (start_w[0], start_w[1])
    bs7._START_TOTAL_KEY = _target_key(start_int)         # SAILIR_SUCCESS_TOTAL=1
    bs7._START_SECTOR = bs7._sector_mask(start_int)       # SAILIR_SECTOR_RANK=1 bucket
    if os.environ.get('SAILIR_SYM_DROP', '0') == '1':
        # DO NOT ENABLE IN PRODUCTION — measured null result (locked 2026-07-11):
        # fired 352x on m1/m2/m3 A/B, shortened nothing, cost worker CPU.
        # See the banner in beam_search_truthcull.py. Kept for retrain-era experiments.
        bs7._init_sym_drop(start_int)                     # same-(r,s) drop detector
    if os.environ.get('SAILIR_STRIP_RAWS', '1') != '0':
        # Keep the ACTION side stripped the same way as the expression. With
        # bs7._STRIP_TOTAL on, the expression drops terms below the start in the
        # FULL total order, so the raws must too -- otherwise every enumerated
        # action keeps regenerating exactly the terms the expression discards.
        # get_raw_equation reads a 3-tuple as the total-ordering threshold, and
        # weight() returns (w0, w1, |abs|-tuple), so the start's weight IS it.
        ibp_env.set_raw_strip_threshold(
            weight(start_int) if bs7._STRIP_TOTAL else start_w12)
        if os.environ.get('SAILIR_SECTOR_RANK', '0') == '1':
            # sector-senior order only: never strip out-of-cone terms — the
            # subsector action filter must see them to reject sector-raising
            # (den-row backward) actions. Under the legacy (r,s) order an
            # out-of-cone sub-weight term IS below the start, so the plain
            # strip remains correct there (bit-identical legacy behavior).
            ibp_env.set_raw_strip_cone(bs7._sector_mask(start_int))
    # Make the anchor cap visible to the training-sample stamp (it arrives as a
    # CLI arg, so the dump cannot otherwise see its value).
    os.environ['SAILIR_IRAWS_KEEP_FIRST'] = str(args.iraws_keep_first)
    target_sector = tuple(get_sector_mask(start_int))
    start_expr = {start_int: 1}

    ckpt_path = None
    if args.checkpoint_path and not args.no_checkpoint:
        ckpt_path = args.checkpoint_path

    if args.verbose:
        print(f'[v7-worker] integral I{list(start_int)} weight={start_w12} '
              f'target_sector={target_sector}', flush=True)
        print(f'[v7-worker] beam_width={args.beam_width} max_steps={args.max_steps} '
              f'n_threads={N_THREADS} n_workers={N_WORKERS} '
              f'SUCCESS_TOTAL={bs7._SUCCESS_TOTAL} ACTION_SELECT={bs7._ACTION_SELECT} '
              f'STRIP_TOTAL={bs7._STRIP_TOTAL} ACTION_SCORE={bs7._ACTION_SCORE} '
              f'SCORE={bs7._SCORE_MODE} BEAM_SORT={os.environ.get("SAILIR_BEAM_SORT","weight")} '
              f'NM_PEN={os.environ.get("SAILIR_NM_PENALTY","0.0")} '
              f'raw_strip_thr={ibp_env.get_raw_strip_threshold()}',
              flush=True)

    beam, best_state = beam_search_v5(
        env, model, start_expr, target_sector, start_w12,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        device=args.device,
        beam_sort=os.environ.get('SAILIR_BEAM_SORT', 'weight'),
        max_actions=int(os.environ.get('SAILIR_MAX_ACTIONS', '900')),
        ckpt_path=ckpt_path,
        ckpt_every=args.checkpoint_interval,
        verbose=True,
        resume_from=args.resume_from,
        tabu=args.tabu,   # --no-tabu was parsed into args.tabu but a
                          # hardcoded True was passed instead, so the
                          # flag silently did nothing
        use_incremental_aux=True,
        use_exprkeyed=False,
        ckpt_every_step=False,
        iraws_window=None,
        iraws_keep_first=args.iraws_keep_first,
        lazy_rs=True,
        model_batch_chunk=8,
        top_k=None,
        n_workers=N_WORKERS,
    )

    elapsed = time.time() - t0
    success = _is_success(best_state, target_sector)
    path = list(best_state.path)
    n_steps = len(path)

    # Full expression (passengers + masters) via path replay, exactly as
    # beam_search_truthcull.main() does for its --output.
    full = replay_full_expr(start_expr, best_state.path, env)
    full_expr = full[0] if full else dict(best_state.expr)

    # ONE-STEP CONTRACT GUARD (2026-07-22): every result term must be STRICTLY
    # BELOW the start in the workers' total order. A violating result (e.g. a
    # den-row applied backward importing a higher-sector integral) banked as
    # success creates ascending substitution edges at the orchestrator — two of
    # those form a cycle that freezes apply_substitutions (g127 incident).
    # Demote to failure rather than bank a poisoned entry.
    if success and full_expr:
        from symmetry_route import tkey as _order_key
        _k0 = _order_key(start_int)
        _viol = [k for k in full_expr if _order_key(k) <= _k0]
        if _viol:
            print(f'[v7-worker] CONTRACT VIOLATION: {len(_viol)} result '
                  f'term(s) at-or-above the start in the total order '
                  f'(e.g. {list(_viol[0])}) — demoting success to FAILURE',
                  flush=True)
            success = False

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result = {
        'success': success,
        'original_integral': start_int,
        'final_expr': full_expr,
        'path': path,
        'restart_offsets': [],
        'steps': n_steps,
        'time': elapsed,
        'prime': args.prime,
        'peak_memory_kb': peak_rss_kb,
        'best_n_non_masters': best_state.n_non_masters,
        # DUAL beam only: WHICH sub-beam produced the winning state. 0 =
        # weight-sorted lane, 1 = prob-sorted lane. Without it the run shows
        # that the dual beam worked but not which rule did the work, which is
        # the point of running two disjoint lanes.
        'solved_lane': getattr(best_state, 'lane', None),
        'best_max_w12': max_w12(best_state.expr, target_sector),
        'start_w12': start_w12,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)

    if args.verbose:
        status = 'SUCCESS' if success else 'INCOMPLETE'
        drops = sum(1 for v in bs7._drop_memo.values() if v) if bs7._drop_memo else 0
        print(f'[v7-worker] {status} in {elapsed:.2f}s path_len={n_steps} '
              f'nm={best_state.n_non_masters} sym_drops={drops} '
              f'peak_rss={peak_rss_kb/1024:.0f}MB '
              f'-> {args.output}', flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
