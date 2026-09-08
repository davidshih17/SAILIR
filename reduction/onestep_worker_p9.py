#!/usr/bin/env python3
"""onestep_worker_v8.py
=================================================================
Hierarchical-reduction worker that runs `greedy_p9.beam_search_v5`
IN-PROCESS — exactly the way onestep_worker_v6.py runs v6, and exactly the way
the verified probes ran `python greedy_p9.py --n-threads 8 --n-workers 8`.

WHY IN-PROCESS (and not a subprocess wrapper):
  A single process means greedy_p9's import-time _cap_incidental_threads()
  runs in THIS process and confines it + its 8 forked enumerate workers to 8
  cores — byte-for-byte the CPU behavior the probes verified (8/8, zero holds).
  There is no pickle round-trip (so no `ModuleNotFoundError: sailir`), no stdout
  buffering (so the .out shows live progress and survives a kill), and the
  orchestrator-format result.pkl is written directly (so reaping just works).

v7 SETTINGS reproduced (== the success-only probe recipe):
  ENV (set BELOW, before importing greedy_p9, because _SUCCESS_TOTAL /
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

# ── v7 toggles: MUST be set before importing greedy_p9 (module-level
#    _SUCCESS_TOTAL/_BEAM_TOTAL/_ACTION_SELECT read os.environ at import). ──
os.environ.setdefault('PYTHONUNBUFFERED', '1')
os.environ.setdefault('MALLOC_MMAP_THRESHOLD_', '67108864')
os.environ['SAILIR_END_OF_STEP_TRIM'] = '1'
# setdefault, not assignment: unset -> '0' exactly as before (pure aggressive
# tabu, no cap), but an explicit SAILIR_TABU_CAP from the submit file now
# survives. MEASURED need: on 1,0,1,0,0,0,5,1,1,1 at depth 4 the tabu filter
# removed 96 of 668 actions and the truth action was among them -- and that
# action is the MODEL'S RANK-1 PICK at p=1.0000. cap>0 re-opens the
# highest-probability blocked actions once swept cap-deep.
os.environ.setdefault('SAILIR_TABU_CAP', '0')
os.environ['SAILIR_STRIP_RAWS'] = '1'
os.environ['SAILIR_PACKED_RS'] = '1'
os.environ['SAILIR_SUCCESS_TOTAL'] = '1'
# SAILIR_BEAM_TOTAL intentionally absent -> (r,s) beam + (r,s) maxweight.

# ── argv injection so greedy_p9's import-time _cap_incidental_threads()
#    sees the production 8/8 config and applies thread-caps + 8-core pin. It
#    peeks sys.argv for --n-threads/--n-workers; our own parser uses
#    parse_known_args() below so these extra flags are ignored. ──
# CPU count from --v7-cpus (default 8). Peeked from sys.argv BEFORE importing
# greedy_p9, because its import-time _cap_incidental_threads() reads the
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
# symmetry-first block below imports sailir/numpy ahead of greedy_p9's
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
# Before paying the greedy_p9/torch import + model load, try the symmetry
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
        rule = canonical_monolithic_rule(I)
        el = time.time() - t0
        if rule is None:
            print(f'[sym-first] survivor after {el:.1f}s — proceeding to '
                  f'beam search', flush=True)
            return
        k0 = tkey(I)
        assert all(tkey(k) > k0 for k in rule), 'sym-first rule not descending'
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

# Import greedy_p9 FIRST. Its module top runs _cap_incidental_threads()
# (thread caps + 8-core affinity pin) BEFORE it (internally) imports numpy/torch,
# so torch's threadpool is created already-pinned to 8 cores.
#
# CRITICAL ORDER: os.sched_setaffinity(0, mask) pins only the CALLING thread and
# any threads created AFTERWARD. If torch (or sailir.classifier, which imports
# torch) is imported BEFORE this, torch spins up its threadpool across ALL cores,
# and the later main-thread-only pin canNOT reclaim those threads -> total CPU
# exceeds RequestCpus -> Condor holds the job. So greedy_p9 (hence its
# torch) must load before any other torch-importing module. This mirrors the
# standalone greedy_p9.py order (_cap_incidental_threads() then import torch).
import greedy_p9 as bs7
from greedy_p9 import (beam_search_v5, replay_full_expr, _is_success,
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
    p.add_argument('--max_steps', type=int, default=10**6)
    p.add_argument('--prime', type=int, default=1009)
    p.add_argument('--device', default='cpu')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--paper-masters-only',
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume-from', default=None)
    # Accepted for CLI parity with onestep_worker_v6 (some ignored — v7 fixes them).
    p.add_argument('--random-target', action='store_true')
    p.add_argument('--dedup-beam-by-content',
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--beam-sort', default='weight')
    p.add_argument('--no-lazy-rs', dest='lazy_rs', action='store_false', default=True)
    p.add_argument('--no-exprkeyed', dest='use_exprkeyed',
                   action='store_false', default=False)
    p.add_argument('--checkpoint-path', default=None)
    p.add_argument('--no-checkpoint', action='store_true')
    p.add_argument('--checkpoint-interval', type=int, default=100)
    p.add_argument('--checkpoint-time-seconds', type=int, default=300)
    # parse_known_args: tolerate the injected --n-threads/--n-workers flags.
    args, _unknown = p.parse_known_args()

    # ── replicate greedy_p9.main()'s setup VERBATIM ──────────────────
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

    # v7 module globals that greedy_p9.main() sets (replicate exactly).
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
    # Oracle mode replays recorded truth actions and never consults the model,
    # so skip loading entirely (the harness passes /dev/null as the checkpoint).
    # Oracle mode does not imply model-free: with a real checkpoint the beam is
    # genuinely model-guided and only the ON-PATH state is forced to the
    # recorded action, which is the realistic beam>1 test. Skip loading only
    # when no checkpoint was supplied (/dev/null).
    _v9_oracle_mode = (bool(os.environ.get('SAILIR_V9_ORACLE'))
                       and args.model_checkpoint in ('/dev/null', 'none', ''))
    if _v9_oracle_mode:
        print('[worker] v9 ORACLE run -- no checkpoint loaded', flush=True)
        model = None
    ck = (None if _v9_oracle_mode else
          torch.load(args.model_checkpoint, map_location='cpu',
                     weights_only=False))
    if not _v9_oracle_mode:
        _variant = (ck.get('args') or {}).get('model_variant')
        if _variant == 'eqact':
            # EQACT: nosubs + the gated resolved-equation branch. The equations
            # are resolved from the LIVE store at search time (eqact_infer.py);
            # there is no precompute for states the beam invents.
            #
            # use_start_target and the two truncation widths MUST come from the
            # checkpoint's own args, not from this process's environment: a
            # T=10 model read with SAILIR_EQ_TERMS=5 would be fed half the
            # equation it was trained on, silently.
            _a = ck.get('args') or {}
            # NO SILENT DEFAULT. These were env vars during training, so older
            # checkpoints do NOT record them (verified: T=10's best_model.pt has
            # neither key). Guessing would feed a T=5 model twice the equation it
            # trained on and look fine -- exactly how the prime=2^31-1 bug
            # survived two months. Checkpoint value wins; else an EXPLICIT env
            # var; else refuse.
            for _k, _e in (('eq_terms', 'SAILIR_EQ_TERMS'),
                           ('expr_terms', 'SAILIR_EXPR_TERMS')):
                if _a.get(_k) is not None:
                    os.environ[_e] = str(_a[_k])
                elif not os.environ.get(_e):
                    raise SystemExit(
                        f'eqact checkpoint does not record {_k} and {_e} is not '
                        f'set. Refusing to guess the truncation width -- it must '
                        f'match training exactly. Set {_e} in the worker to the '
                        f'value its training script used.')
            import sys as _sys
            _tr = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/training'
            if _tr not in _sys.path:
                _sys.path.insert(0, _tr)
            from sailir.classifier_eqact import IBPActionClassifierEqAct
            model = IBPActionClassifierEqAct(
                prime=_a.get('prime', args.prime),
                n_indices=topology.n_indices,
                n_denominators=topology.n_denominators,
                n_ibp_ops=topology.n_actions,
                use_start_target=bool(_a.get('use_start_target', False)),
            )
            print(f'[worker] eqact model: use_start_target='
                  f'{bool(_a.get("use_start_target", False))} '
                  f'EQ_TERMS={os.environ["SAILIR_EQ_TERMS"]} '
                  f'EXPR_TERMS={os.environ["SAILIR_EXPR_TERMS"]}', flush=True)
        elif _variant == 'nosubs':
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
        # See the banner in greedy_p9.py. Kept for retrain-era experiments.
        bs7._init_sym_drop(start_int)                     # same-(r,s) drop detector
    if os.environ.get('SAILIR_STRIP_RAWS', '1') != '0':
        # Keep the ACTION side stripped the same way as the expression. With
        # bs7._STRIP_TOTAL on, the expression drops terms below the start in the
        # FULL total order, so the raws must too -- otherwise every enumerated
        # action keeps regenerating exactly the terms the expression discards.
        # get_raw_equation reads a 3-tuple as the total-ordering threshold, and
        # weight() returns (w0, w1, |abs|-tuple), so the start's weight IS it.
        ibp_env.set_raw_strip_threshold(
            weight(start_int))
        if os.environ.get('SAILIR_SECTOR_RANK', '0') == '1':
            # sector-senior order only: never strip out-of-cone terms — the
            # subsector action filter must see them to reject sector-raising
            # (den-row backward) actions. Under the legacy (r,s) order an
            # out-of-cone sub-weight term IS below the start, so the plain
            # strip remains correct there (bit-identical legacy behavior).
            ibp_env.set_raw_strip_cone(bs7._sector_mask(start_int))
    target_sector = tuple(get_sector_mask(start_int))
    start_expr = {start_int: 1}

    ckpt_path = None
    if args.checkpoint_path and not args.no_checkpoint:
        ckpt_path = args.checkpoint_path

    if args.verbose:
        print(f'[v7-worker] integral I{list(start_int)} weight={start_w12} '
              f'target_sector={target_sector}', flush=True)
        print(f'[v7-worker] max_steps={args.max_steps} '
              f'n_threads={N_THREADS} '
              f'SUCCESS_TOTAL={bs7._SUCCESS_TOTAL} '
              f'BEAM_SORT={os.environ.get("SAILIR_BEAM_SORT","weight")} '
              f'NM_PEN={os.environ.get("SAILIR_NM_PENALTY","0.0")} '
              f'raw_strip_thr={ibp_env.get_raw_strip_threshold()}',
              flush=True)

    beam, best_state = beam_search_v5(
        env, model, start_expr, target_sector, start_w12,
        max_steps=args.max_steps,
        device=args.device,
        ckpt_path=ckpt_path,
        ckpt_every=args.checkpoint_interval,
        verbose=True,
        resume_from=args.resume_from,
        use_incremental_aux=True,
        use_exprkeyed=False,
        ckpt_every_step=False,
        lazy_rs=True,
        model_batch_chunk=8,
    )

    elapsed = time.time() - t0
    success = _is_success(best_state, target_sector)
    path = list(best_state.path)
    n_steps = len(path)

    # Full expression (passengers + masters) via path replay, exactly as
    # greedy_p9.main() does for its --output.
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
