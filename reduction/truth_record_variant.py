#!/usr/bin/env python
"""EXPERIMENT VARIANT of truth_record_training_v2.py -- do NOT use for corpus generation.

Answers two questions the production recorder cannot:
  (a) HOW MANY unused closure rows could solve the current target? The
      production loop takes the lowest row index and never counts the rest,
      so we do not know whether its label is forced or one of many.
  (b) Do the OTHER rows also lead to a valid one-step reduction? Selected
      via SAILIR_TRUTH_PICK = first (default, == production) | last | random.

If "last" and "random" also reach success=True, the recorded label is an
arbitrary choice among equivalents, and cross-entropy against it is training
the model to imitate a tie-break rather than to reduce.

SAILIR_TRUTH_PICK also accepts CANONICAL rules -- minterms | minsumw |
minmaxw (and maxterms as a deliberately-bad control) -- which choose the row
by a recognisable property of the substitution it produces, measured on the
ACTIVE tail only. That makes the label a deterministic function of the state,
hence learnable, and lets the beam sort key be aligned with the same cost.

SAILIR_TRUTH_STRIP defaults to 1: work modulo lower sectors and lower total
lex weight, as the worker does. Set it to 0 for the unstripped reference run.
NOTE: under stripping the end-state verification is vacuous and is SKIPPED --
soundness comes from trajectory equivalence against an unstripped run
(results/truth/strip/compare.py).

SAILIR_TRUTH_SEED seeds the random pick.
"""
_ORIG_DOC = """TRUTH -> TRAINING DATA v2 (2026-08-01): DATA-GEN-FAITHFUL recording.

v1 (truth_record_training.py) deviated from the data-gen pipeline by
materializing rule dependencies into the substitution store — producing
states (1000-entry subs, 30k-action enumerations) that the generator never
creates and the trainer format rejects. RETIRED.

v2 changes as little as possible relative to data-gen:
  used_ibps := the truth engine's one-step action closure for the target
               (worker_replay: primitive (op, seed) actions, same shape as
               a recorded scramble)
  then the generator's unscramble loop runs VERBATIM (same target selection,
  same one-entry-per-step subs growth, same enumerate_valid_actions, same
  action-validity gate, samples in the identical schema), with two one-step
  adaptations:
    - termination: stop when no in-sector non-master remains at-or-above the
      start target in the workers' total order (bucket drained), instead of
      reduce-to-masters
    - verification: the surviving expression must EXACTLY equal the truth
      engine's one-step result (replaces the master-coefficient check)
  A trajectory that fails (no usable action / action not enumerable /
  end-state mismatch) is DROPPED and counted, exactly like data-gen.

Usage: SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1 \
  truth_record_training_v2.py --integral 2,1,1,... --output out.jsonl
Prints one STATS line (success/failure + counts) per invocation.
"""
import os
import random
import sys
import json
import time
import argparse
import numpy as np

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))
sys.path.insert(0, os.path.join(ROOT, "data-gen"))

from sailir.topology import Topology
import topo_config as _tc
import generate_multisector_data as gm
import truth_engine as _te
from truth_engine import TruthEngine, sector_of, rs_of
from symmetry_route import tkey

# The field. This was hardcoded to 1009 while truth_engine.P already read
# SAILIR_PRIME, so pointing the recorder at a p=101 corpus would have had the
# two halves of the same run working over DIFFERENT fields -- silently, since
# every coefficient stays a small integer either way. Read the same variable,
# and assert they agree rather than trust that they do.
P = int(os.environ.get('SAILIR_PRIME', '1009'))
assert P == _te.P, (f'prime mismatch: recorder P={P} but truth_engine.P='
                    f'{_te.P}; SAILIR_PRIME must be set before import')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--integral', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--dr', type=int, default=1)
    ap.add_argument('--ds', type=int, default=1)
    args = ap.parse_args()

    # Validate the pick rule BEFORE the truth phase: that phase costs 107s on
    # the cheapest of these integrals and 5000s on the most expensive, so a
    # typo must not be discovered after paying for it.
    _PICK = os.environ.get('SAILIR_TRUTH_PICK', 'first')
    _CANON = ('minterms', 'maxterms', 'minsumw', 'minmaxw',
              'minnew', 'minnewsumw')
    if _PICK not in _CANON + ('first', 'last', 'random', 'lex', 'model'):
        raise SystemExit(f"unknown SAILIR_TRUTH_PICK={_PICK!r}")

    T = tuple(int(x) for x in args.integral.strip("'\"").split(","))
    S0 = sector_of(T)
    r0, s0 = rs_of(T)

    topology = Topology.from_dir(_tc.TOPO_DIR)
    gm.init_from_topology(topology)
    gm.PRIME = P
    ibp_t = gm.parse_templates(os.path.join(_tc.TOPO_DIR, 'IBP'))
    li_t = gm.parse_templates(os.path.join(_tc.TOPO_DIR, 'LI'))
    n_ibp = len(ibp_t)
    shifts = {}
    for op in range(n_ibp):
        if op in ibp_t:
            shifts[op] = [sh for sh, _ in ibp_t[op]]
    for li_idx in li_t:
        shifts[n_ibp + li_idx] = [sh for sh, _ in li_t[li_idx]]

    # PERF (2026-08-01, evidence: live stack sampling shows eval() in
    # evaluate_coefficient dominating): memoize raw-equation generation —
    # enumeration regenerates identical (op, seed) equations from coefficient
    # STRINGS (Python eval + Fraction) hundreds of times per step. Bounded
    # cache; deterministic, so values are unchanged.
    _raw_memo = {}
    _orig_raw = gm.get_raw_equation
    raw_stat = dict(calls=0, time=0.0)

    def _timed_raw(it_, lt_, op_, seed_):
        _t = time.time()
        eq = _orig_raw(it_, lt_, op_, seed_)
        raw_stat['calls'] += 1
        raw_stat['time'] += time.time() - _t
        return eq

    def _memo_raw(it_, lt_, op_, seed_):
        k = (op_, tuple(seed_))
        eq = _raw_memo.get(k)
        if eq is None:
            eq = _timed_raw(it_, lt_, op_, seed_)
            if len(_raw_memo) > 1_000_000:
                _raw_memo.clear()
            _raw_memo[k] = eq
        return eq

    if os.environ.get('SAILIR_RAW_MEMO', '1') != '0':
        gm.get_raw_equation = _memo_raw
    else:
        gm.get_raw_equation = _timed_raw

    t0 = time.time()
    # SAILIR_LOAD_CLOSURE=<path>: reuse a closure computed EARLIER (by a run
    # with SAILIR_DUMP_CLOSURE, or by the batched closure campaign) instead of
    # recomputing it. worker_replay is the whole expense of this program --
    # median 0.13 s but a p90 of 34 s and a tail to 4 h -- and the closure it
    # returns is a state-independent property of the target, so recomputing it
    # once per corpus variant is pure waste.
    #
    # ONLY the (op, seed) list is reused. `truth_result` cannot be: the dump
    # does not carry it. That is safe under SAILIR_TRUTH_STRIP=1 (the default,
    # and what every corpus campaign uses) because end-state verification is
    # SKIPPED in strip mode -- see the _STRIP branch below, where the comparison
    # against truth_result is unreachable. Refuse to run unstripped, rather than
    # compare against None and report a spurious mismatch.
    _load = os.environ.get('SAILIR_LOAD_CLOSURE')
    if _load:
        if os.environ.get('SAILIR_TRUTH_STRIP', '1') != '1':
            print('[closure] SAILIR_LOAD_CLOSURE requires SAILIR_TRUTH_STRIP=1 '
                  '(no truth_result in the dump to verify against)', flush=True)
            return 2
        _cd = json.load(open(_load))
        if tuple(_cd['integral']) != T:
            print(f"[closure] LOADED CLOSURE IS FOR A DIFFERENT INTEGRAL: "
                  f"{_cd['integral']} != {list(T)}", flush=True)
            return 2
        used_ibps = [(int(_op), tuple(_sd)) for _op, _sd in _cd['closure']]
        truth_result = None
        print(f'[closure] loaded {len(used_ibps)} rows <- {_load}', flush=True)
    # The ENGINE is constructed either way. Its __init__ is the only thing that
    # configures sailir.ibp_env -- init_from_topology, set_prime,
    # set_paper_masters_only(True) and apply_canonical_masters() under
    # SAILIR_SECTOR_RANK=1 -- so skipping it would leave that global state unset
    # and silently change what counts as a master. It costs ~0.1 s; the expense
    # is worker_replay, and that is what the load path actually skips.
    eng = TruthEngine(_tc.TOPO_DIR, os.path.join(
        _tc.TOPO_DIR, "kira_validate/sectormappings/GR/trivialsector"))
    if not _load:
        wr = eng.worker_replay(T, dr=args.dr, ds=args.ds, verbose=True)
        # the truth closure as a scramble-style action record + the exact result
        used_ibps = [(op, tuple(x + d for x, d in zip(J, delta)))
                     for (J, op, delta) in wr['actions']]
        truth_result = {k: v % P for k, v in wr['result_expr'].items()}
    # SAILIR_DUMP_CLOSURE=<path>: write the truth engine's closure library --
    # the state-independent SET of (op, seed) rows -- so a worker-driven walk can
    # restrict its post-cull candidates to rows known to reduce. The recording's
    # `chosen_action` sequence is NOT enough: it is one path, and any cull that
    # changes a pick leaves it immediately. Written before the replay loop so it
    # costs only the truth phase.
    _dump = os.environ.get('SAILIR_DUMP_CLOSURE')
    if _dump:
        with open(_dump, 'w') as _df:
            json.dump({'integral': list(T), 'sector': S0,
                       'closure': [[int(_op), list(_sd)]
                                   for _op, _sd in used_ibps]}, _df)
        print(f'[closure] wrote {len(used_ibps)} rows -> {_dump}', flush=True)
    t_truth = time.time() - t0
    print(f"[plan] closure={len(used_ibps)} actions -> step budget "
          f"~{len(used_ibps)} (hard cap {len(used_ibps) + 500}); "
          f"truth phase {t_truth:.1f}s", flush=True)

    # ---- the generator's unscramble loop, adapted only as documented ----
    # PERF (2026-08-01): each unused action's SUBSTITUTED equation is kept
    # incrementally current (rows updated only when a newly solved integral
    # appears in them) instead of recomputed from raw+store per step per
    # candidate — same math (apply_all_substitutions is order-independent),
    # turns the O(actions x store) per-step scan into O(affected rows).
    kT = tkey(T)
    # ---- CORRUPTION INJECTION CONFIG (SAILIR_CORRUPT_FRAC) ----------------
    # OFF by default (0.0): with _CFRAC == 0 not a single line of the injection
    # block below executes, so an uncorrupted run is bit-identical to the
    # existing campaign and the two corpora stay comparable.
    _CFRAC = float(os.environ.get('SAILIR_CORRUPT_FRAC', '0') or 0)
    _CMIN = int(os.environ.get('SAILIR_CORRUPT_MIN', '1'))
    _CMAX = int(os.environ.get('SAILIR_CORRUPT_MAX', '3'))
    _CSEED = int(os.environ.get('SAILIR_CORRUPT_SEED', '0'))
    _crng = random.Random(_CSEED)
    n_injected = 0
    _n_inactive_intro = [0]
    _n_intro = [0]
    _n_reintro = [0]
    _n_attempt = [0]
    _n_ok = [0]
    _n_skipped = [0]
    _n_noop = [0]
    _ns_ok = {}
    _ns_rej = {}
    _ns_used = {}
    # ---- CANDIDATE CULL (SAILIR_CULL_M) ----------------------------------
    # OFF by default (-1): _cull() returns its argument unchanged, so the only
    # added work is one dict write per step and the walk is bit-identical.
    #
    # WHY A CULL IS POSSIBLE AT ALL. enumerate_valid_actions has two loops
    # (sailir/ibp_env.py:1039). The DIRECT loop runs over `shifts`, a fixed
    # table built once from the topology, so it contributes a bounded number of
    # candidates forever -- 332 distinct (op, delta) pairs here. The INDIRECT
    # loop runs `for sub_int in subs`, and subs gains exactly one entry per
    # step (subs[target] = sol below), so indirect candidates grow LINEARLY
    # with the step number. Measured: ~26,650 candidates/step averaged over a
    # 1,459-step reduction, of which 0.56% are direct.
    #
    # THE RULE. Keep every direct candidate, plus indirect candidates whose
    # source substitution was added within the last M steps. Any finite M turns
    # linear growth into a CONSTANT ceiling, which is the whole point: the
    # longer the reduction, the larger the saving.
    #
    # RECOVERING THE SOURCE. An indirect action against sub_int uses
    # seed = sub_int - shift, so sub_int = seed + shift. Testing each shift of
    # that row's op against the substitution keys recovers it. A row reachable
    # from several substitutions is credited to the MOST RECENT, since that is
    # the one that keeps it alive under the cull.
    # ADAPTIVE WINDOW (SAILIR_CULL_ALPHA). A fixed M gives a constant ceiling
    # while the full action space grows linearly, so the kept FRACTION shrinks
    # to nothing on a long walk -- and the per-step block rate is what kills a
    # reduction (a 0.94% block over 375 steps is a 97% chance of death).
    #
    # Keeping a fixed FRACTION of the total valid space instead is, to a good
    # approximation, a window that grows with the step number: the number of
    # indirect actions sourced from any one substitution is roughly constant
    # (~35 measured), so "the newest alpha x |valid| actions" ~ "the last
    # alpha x t steps". That needs no enumeration -- just a step-dependent M.
    #
    # M_t = max(SAILIR_CULL_M, alpha * t). SAILIR_CULL_M is then a FLOOR that
    # protects the early steps, where alpha*t is tiny.
    _SKIP_ENUM = os.environ.get('SAILIR_SKIP_ENUM', '0') == '1'
    if _SKIP_ENUM and _PICK == 'model':
        raise SystemExit("SAILIR_SKIP_ENUM=1 is incompatible with "
                         "SAILIR_TRUTH_PICK=model: the model scores the "
                         "enumerated action list, so there is nothing to rank.")
    _CULL_M = int(os.environ.get('SAILIR_CULL_M', '-1'))
    _CULL_ALPHA = float(os.environ.get('SAILIR_CULL_ALPHA', '0') or 0)

    def _M_at(it):
        return (max(_CULL_M, int(_CULL_ALPHA * it)) if _CULL_ALPHA > 0
                else _CULL_M)
    _DIRECT = frozenset(
        (_op, tuple(-_s[_i] for _i in range(gm.N_INDICES)))
        for _op, _lst in shifts.items() for _s in _lst)
    _sub_step = {}          # substitution key -> the step it was added
    _cull_kept = [0]
    _cull_full = [0]
    _cull_lost_lex = [0]    # steps where the cull removed the UNCULLED pick
    _need_hist = {}         # smallest M that would have worked, per step
    _RELAPORTA = os.environ.get('SAILIR_RELAPORTA', '0') == '1'
    _RELAP_MAX = int(os.environ.get('SAILIR_RELAPORTA_MAX', '50'))
    _n_relap = [0]
    _n_relap_fail = [0]
    _n_direct_rescue = [0]
    _t_relap = [0.0]
    _seen_rows = set(used_ibps)   # (op, seed) already in the library

    def _cull(cands, tgt, it):
        """Filter to S_n & T_M, and record the smallest M that would have
        sufficed at this step.

        THE SECOND RETURN IS THE POINT. The required M is a property of the
        TRAJECTORY, and each pick rule walks a different trajectory -- so the
        505 measured on the lex-walked recordings says nothing about what
        minsumw or minmaxw need. Computing it live, along the walk the rule
        actually takes, answers that in one run instead of bisecting over M.

        It is the age of the YOUNGEST candidate (0 if any is direct). If the
        max of this over a SUCCESSFUL walk comes out well below the M that was
        used, that M was more generous than the rule needed.
        """
        if _CULL_M < 0 or not cands:
            return cands, None
        keep = []
        _need = None
        for _ci in cands:
            _op, _seed = used_ibps[_ci]
            if (_op, tuple(_seed[_i] - tgt[_i]
                           for _i in range(gm.N_INDICES))) in _DIRECT:
                keep.append(_ci)
                _need = 0
                continue
            _newest = None
            for _s in shifts.get(_op, ()):
                _j = _sub_step.get(tuple(_seed[_i] + _s[_i]
                                         for _i in range(gm.N_INDICES)))
                if _j is not None and (_newest is None or _j > _newest):
                    _newest = _j
            if _newest is None:
                continue
            _age = it - _newest
            if _need is None or _age < _need:
                _need = _age
            if _age <= _M_at(it):
                keep.append(_ci)
        return keep, _need

    expr = {T: 1}
    _rng = random.Random(int(os.environ.get('SAILIR_TRUTH_SEED', '0')))
    cand_counts = []
    n_labels_hist = []
    canon_cost = []
    model_trace = []
    _mdl = {}

    def _model_probs(expr_f, subs_, valid_, secmask_, tgt_):
        # Lazy so that no non-model pick pays the torch import or the load.
        # Scoring mirrors rank_on_recording.py exactly (same batch prep, same
        # forward, same slice) so numbers are comparable between the two.
        if not _mdl:
            import torch
            from sailir.classifier_nosubs import IBPActionClassifierNoSubs
            _ck = torch.load(os.environ['SAILIR_MODEL_CKPT'],
                             map_location='cpu', weights_only=False)
            _a = _ck.get('args') or {}
            if hasattr(_a, '__dict__'):
                _a = vars(_a)
            _m = IBPActionClassifierNoSubs(
                embed_dim=_a['embed_dim'], n_heads=_a['n_heads'],
                n_expr_layers=_a['n_expr_layers'],
                n_cross_layers=_a['n_cross_layers'],
                prime=_a['prime'], n_indices=topology.n_indices,
                n_denominators=topology.n_denominators,
                n_ibp_ops=topology.n_actions)
            _m.load_state_dict(_ck['model_state_dict'])
            _m.eval()
            sys.argv = [sys.argv[0], '--v7-cpus', '1']
            import beam_search_v7 as _bs7
            _mdl['m'] = _m
            _mdl['bs7'] = _bs7
            _mdl['torch'] = torch
            print(f"[model] {os.environ['SAILIR_MODEL_CKPT']} "
                  f"epoch={_ck.get('epoch')} prime={_a['prime']}", flush=True)
        _b = _mdl['bs7'].prepare_batched_input_v5_dummy(
            [(expr_f, subs_, valid_, secmask_, tgt_)], 'cpu',
            max_actions=len(valid_))
        with _mdl['torch'].no_grad():
            _, _pp = _mdl['m'](
                _b['expr_integrals'], _b['expr_coeffs'], _b['expr_mask'],
                _b['sub_keys'], _b['sub_repl_ints'], _b['sub_repl_coeffs'],
                _b['sub_repl_mask'], _b['sub_mask'],
                _b['action_ibp_ops'], _b['action_deltas'], _b['action_mask'],
                _b['sector_mask'], _b['target_integral'])
        return _pp[0, :len(valid_)].numpy()
    # Canonical rules: the label becomes a deterministic function of the state.
    # 'maxterms' is a deliberately-bad control -- if it closes with a path
    # length similar to 'minterms', the cost does not matter and the whole
    # canonicalisation buys nothing but learnability.
    print(f"[variant] pick strategy = {_PICK}", flush=True)
    subs = {}
    # Incremental enum cache (2026-08-04): keeps
    # apply_all_substitutions(raw(op,seed), subs) for the indirect loop alive
    # across steps instead of replaying the whole substitution history for
    # every candidate action at every step. SAILIR_ENUM_CACHE=0 disables.
    enum_cache = (gm.EnumCache()
                  if os.environ.get('SAILIR_ENUM_CACHE', '1') == '1' else None)
    used_set = set()
    samples = []
    success = True
    failure_reason = None
    ph = dict(search=0.0, enum=0.0, apply=0.0)
    # ---- STRIP MODE (SAILIR_TRUTH_STRIP=1) --------------------------------
    # Work modulo lower sectors and lower total lex weight, as the worker does.
    # A term outside the active set is a passenger: it is never a target, never
    # needs reducing, and is recovered by replay. Dropping it is SAFE because
    # the active set is defined by a FIXED T -- once a term is below T or in a
    # lower sector it can never come back above, so the predicate is monotone.
    #
    # Payoff is twofold. Speed: rows and substitution tails carry only active
    # terms, so enumeration (537-828s of the ~800-950s runtime) shrinks. And
    # FIDELITY: the recorder currently writes filter_sector_only(expr), which
    # drops lower sectors but KEEPS terms below T, while the worker strips
    # both -- so training states carry passengers the worker's states never
    # have. Stripping removes that train/inference mismatch.
    #
    # Cost: the end-state verification compares the final expression against
    # the truth engine's full result. Under stripping the final active state is
    # empty by construction, so that check becomes vacuous and is skipped.
    # Validation is instead by trajectory equivalence against an unstripped run
    # (see canon/strip_compare.py) -- identical action sequence == sound.
    # DEFAULT ON. Set SAILIR_TRUTH_STRIP=0 to opt out (needed for the
    # unstripped reference run that validates stripping, and for any run whose
    # end-state verification must actually test something).
    _STRIP = os.environ.get('SAILIR_TRUTH_STRIP', '1') == '1'

    def _active(k):
        return sector_of(k) == S0 and tkey(k) <= kT

    def _strip(d):
        return {k: v for k, v in d.items() if _active(k)} if _STRIP else d

    cached_rows = {}
    occur = {}
    _n_truth_rows = len(used_ibps)   # rows beyond this index are injected                    # integral -> set of row idxs containing it
    for idx, (ibp_op, seed) in enumerate(used_ibps):
        row = _strip(dict(gm.get_raw_equation(ibp_t, li_t, ibp_op, seed)))
        cached_rows[idx] = row
        for k in row:
            occur.setdefault(k, set()).add(idx)

    def _apply_new_sub(t_solved, sol):
        for ridx in list(occur.get(t_solved, ())):
            row = cached_rows.get(ridx)
            if row is None or t_solved not in row:
                continue
            co = row.pop(t_solved)
            occur[t_solved].discard(ridx)
            for K, cK in sol.items():
                v = (row.get(K, 0) + co * cK) % P
                if v:
                    if K not in row:
                        occur.setdefault(K, set()).add(ridx)
                    row[K] = v
                else:
                    if K in row:
                        occur.get(K, set()).discard(ridx)
                    row.pop(K, None)
    # Each injection adds up to _CMAX identities and each needs a step to undo,
    # so the budget must grow with the corruption rate or long trajectories run
    # out of iterations before the closure drains.
    _cap = len(used_ibps) + 500
    if _CFRAC:
        _cap += int(len(used_ibps) * _CFRAC * _CMAX * 4) + 500
    for iteration in range(_cap):
        # ---- INJECT A CORRUPTION -------------------------------------
        # Scramble identities go into used_ibps / cached_rows / occur -- the
        # SAME pools the truth closure rows live in. That is the entire design:
        # the loop below picks the heaviest active non-master and then any
        # UNUSED row acting on it, so a scramble identity is found by the same
        # search, ranked by the same lex key, and emitted through the same code
        # path with the same schema. Nothing downstream can distinguish an
        # unscramble step from a truth step -- which is precisely what the
        # worker sees, and why there is no separate corpus and no marker field.
        if _CFRAC and _crng.random() < _CFRAC:
            # NO GUARD ON WHAT THE SCRAMBLE INTRODUCES. Both checks that used
            # to be here were wrong:
            #
            #  - "re-introduces an already-solved integral": handled by folding
            #    the accumulated store through the scrambled expression below,
            #    which is what the worker does anyway. Rejecting these cost 30
            #    of 49 draws and left 1 in 50 usable.
            #  - "introduces terms outside the active set": scramble cannot
            #    reach a supersector, and subsectors and lighter integrals are
            #    discarded as passengers by design. That is normal operation,
            #    not a failure.
            #
            # A draw is only skipped if it changes nothing after the fold.
            _n_attempt[0] += 1
            _prev_expr = dict(expr)
            for _try in range(6):
                _ns = _crng.randint(_CMIN, _CMAX)
                # REPRODUCIBILITY. gm.scramble draws from the GLOBAL random
                # module (random.shuffle / random.choices / random.choice) and
                # nothing seeds it, so the corruption differed run to run:
                # two identical configurations gave 86 and 84 steps with the
                # same depth histogram. Seed it per (target, step, retry) so a
                # rebuild reproduces the corpus exactly. Only reached when
                # corruption is enabled, so the default path is untouched.
                random.seed((_CSEED * 1000003 + iteration * 9176 + _try) & 0x7fffffff)
                try:
                    _sc, _new = gm.scramble(expr, ibp_t, li_t, len(shifts), _ns,
                                            S0, filter_lateral=False)
                except Exception:
                    _sc, _new = None, []
                if not _new or _sc is None or _sc == expr:
                    continue
                _intro = [k for k in _sc if k not in expr]
                # RESTORE THE INVARIANT. expr is always fully reduced with
                # respect to subs -- the loop applies each substitution as it
                # is made. scramble() breaks that by injecting integrals
                # without folding the store through the result. Folding it here
                # is exactly what the worker does, and it means a corruption
                # that reintroduces an already-solved integral is simply
                # re-expanded on the spot instead of stalling the walk. That
                # removes the need to reject such draws at all: previously 30
                # of 49 rejections were for this, leaving 1 draw in 50 usable.
                expr = _strip(gm.apply_all_substitutions(dict(_sc), subs))
                if expr == _prev_expr:
                    _n_noop[0] += 1
                    continue
                for _op, _seed in _new:
                    _ri = len(used_ibps)
                    used_ibps.append((_op, _seed))
                    _row = gm.apply_all_substitutions(
                        _strip(dict(gm.get_raw_equation(ibp_t, li_t,
                                                        _op, _seed))), subs)
                    cached_rows[_ri] = _row
                    for _k in _row:
                        occur.setdefault(_k, set()).add(_ri)
                n_injected += len(_new)
                _n_intro[0] += len(_intro)
                _n_ok[0] += 1
                # depth of the ACCEPTED draw, and of every rejected one, so the
                # guard's bias toward shallow scrambles is measured rather than
                # inferred: a deeper draw introduces more terms, so it has more
                # chances to hit a solved integral or leave the active set, and
                # the guard rejects the whole draw if ANY term violates.
                _ns_ok[_ns] = _ns_ok.get(_ns, 0) + 1
                _ns_used[len(_new)] = _ns_used.get(len(_new), 0) + 1
                break
            else:
                _n_skipped[0] += 1

        # one-step termination: bucket = in-sector non-masters at-or-above T
        bucket = {k: v for k, v in expr.items()
                  if sector_of(k) == S0
                  and not gm.is_master_for_sector(k, S0)
                  and tkey(k) <= kT}
        if not bucket:
            # verification: replay end state and truth result are different
            # compositions of the same identity — compare NORMAL FORMS (fold
            # both through the engine's rule store until no rule applies)
            final = {k: v for k, v in expr.items() if v}
            _rules = {}
            for _sr in eng.systems.values():
                for _piv, _tup in _sr.items():
                    _rules[_piv] = _tup[0]

            def _nf(e):
                e = dict(e)
                while True:
                    hit = next((k for k in e if k in _rules), None)
                    if hit is None:
                        return e
                    co = e.pop(hit)
                    for K, cK in _rules[hit].items():
                        v = (e.get(K, 0) + co * cK) % P
                        if v:
                            e[K] = v
                        else:
                            e.pop(K, None)

            if _STRIP:
                # Vacuous under stripping: every term the comparison would
                # test has been dropped as a passenger, so `final` is active-
                # only (and empty at the close) while truth_result is the full
                # identity. Say so rather than reporting a pass that tested
                # nothing. Soundness is established out-of-band by trajectory
                # equivalence against an unstripped run.
                failure_reason = "verification_skipped_strip_mode"
                print("[variant] STRIP MODE: end-state verification SKIPPED "
                      "(final active state is empty by construction). "
                      "Validate by comparing the action sequence against an "
                      "unstripped run.", flush=True)
            elif _nf(final) != _nf(truth_result):
                success = False
                failure_reason = (f"end_state_mismatch_normalform: "
                                  f"{len(final)} vs {len(truth_result)} terms")
            break

        # TARGET BY THE TOTAL ORDER, which is (SECTOR RANK, r, s, |abs|).
        # This used to sort by (-w0, -w1, w2) -- i.e. (r, s, |abs|) with NO
        # sector-rank prefix -- so it walked a different target sequence from
        # the worker, whose _target_key carries the rank prefix under
        # SAILIR_SECTOR_RANK=1. Same rule, different states, and a closure
        # recorded here did not cover the worker's walk. tkey IS the total
        # order (smaller = higher), the same key the strip already uses via
        # `tkey(k) <= kT`, so target selection and strip now agree.
        target = min(bucket.keys(), key=tkey)

        _tp = time.time()
        # VARIANT: collect ALL usable rows, not just the first, so the size of
        # the choice set is measurable and an alternative can be selected.
        _cands = [idx for idx in sorted(occur.get(target, ()))
                  if idx not in used_set and cached_rows[idx].get(target, 0) != 0]
        # CULL BEFORE THE PICK, so the rule is "lowest lex action SURVIVING the
        # cull" rather than "lowest lex action, then check it survived".
        _n_full = len(_cands)
        _lex_full = None
        if _CULL_M >= 0 and _cands:
            _lex_full = min(_cands, key=lambda _c: (
                used_ibps[_c][0],
                tuple(used_ibps[_c][1][_i] - target[_i]
                      for _i in range(gm.N_INDICES)), _c))
            _cands, _need_m = _cull(_cands, target, iteration)
            if _need_m is not None:
                _need_hist[_need_m] = _need_hist.get(_need_m, 0) + 1
            _cull_full[0] += _n_full
            _cull_kept[0] += len(_cands)
            if _lex_full not in _cands:
                _cull_lost_lex[0] += 1
            if not _cands and _RELAPORTA and _n_relap[0] < _RELAP_MAX:
                # RE-LAPORTA ON A DEAD END.
                #
                # A dead end never means "nothing can act on the target":
                # DIRECT actions are never culled, so T_M always holds ~148 of
                # them. It means the truth engine's CLOSURE LIBRARY has no
                # direct row for this target and none of its rows are recent
                # enough. The direct actions that would work are simply not in
                # the library.
                #
                # So ask the truth engine for a closure of THIS target. A
                # closure built for X necessarily contains a row acting on X
                # directly, and direct rows survive any cull -- so after this
                # merge the step cannot be blocked, by construction.
                # STEP 1 -- TAKE A DIRECT ACTION, CHOSEN TO RAISE THE MAX
                # WEIGHT LEAST. Direct actions need no Laporta at all: they are
                # seed = target - shift over the fixed 332-entry table, always
                # available, one pass to build. The library not containing them
                # is exactly what makes them an escape from a dead end.
                #
                # Picking by min-max-weight is what makes STEP 2 affordable:
                # the rebuild that follows is for the NEW heaviest integral, so
                # keeping that as light as possible is what holds the seed count
                # inside the ladder's budget.
                _t_rl = time.time()
                _best_d = None
                for _op2, _lst2 in shifts.items():
                    for _sh2 in _lst2:
                        _sd2 = tuple(target[_i] - _sh2[_i]
                                     for _i in range(gm.N_INDICES))
                        if (_op2, _sd2) in _seen_rows:
                            continue
                        _raw2 = gm.get_raw_equation(ibp_t, li_t, _op2, _sd2)
                        if target not in _raw2 or _raw2[target] == 0:
                            continue
                        _row2 = gm.apply_all_substitutions(
                            _strip(dict(_raw2)), subs)
                        if _row2.get(target, 0) == 0:
                            continue
                        _act2 = [_k for _k in _row2
                                 if _k != target and sector_of(_k) == S0
                                 and not gm.is_master_for_sector(_k, S0)
                                 and tkey(_k) <= kT]
                        _c2 = max((gm.weight(_k)[:2] for _k in _act2),
                                  default=(0, 0))
                        _key2 = (_c2, _op2, _sd2)
                        if _best_d is None or _key2 < _best_d[0]:
                            _best_d = (_key2, _op2, _sd2, _row2)
                if _best_d is not None:
                    _, _op2, _sd2, _row2 = _best_d
                    _seen_rows.add((_op2, _sd2))
                    _ri2 = len(used_ibps)
                    used_ibps.append((_op2, _sd2))
                    cached_rows[_ri2] = _row2
                    for _k in _row2:
                        occur.setdefault(_k, set()).add(_ri2)
                    _n_direct_rescue[0] += 1
                    print(f"[rescue] it={iteration} direct action op={_op2} "
                          f"maxw={_best_d[0][0]} added", flush=True)

                # STEP 2 -- REBUILD. Off the truth trajectory now, so the
                # library no longer describes the state; ask the truth engine
                # for identities covering the NEW heaviest active integral.
                _n_relap[0] += 1
                try:
                    _rt = target
                    if _best_d is not None:
                        _candk = [_k for _k in _best_d[3]
                                  if _k != target and sector_of(_k) == S0
                                  and not gm.is_master_for_sector(_k, S0)
                                  and tkey(_k) <= kT]
                        if _candk:
                            _rt = max(_candk, key=gm.weight)
                    _wr2 = eng.worker_replay(_rt, dr=args.dr, ds=args.ds,
                                             verbose=False)
                except (RuntimeError, MemoryError) as _e:
                    # THE REBUILD IS NOT FREE. worker_replay climbs a ladder of
                    # seed systems and SKIPS any rung above 25,000 seeds
                    # (truth_engine.py:381). A mid-reduction target is heavier
                    # than the ladder was budgeted for, so the rebuild that
                    # would rescue a dead end is exactly the Laporta cost the
                    # cull was meant to avoid. Record the rate instead of
                    # crashing -- how OFTEN it is unavailable is the result.
                    _t_relap[0] += time.time() - _t_rl
                    _n_relap_fail[0] += 1
                    print(f"[relaporta] it={iteration} UNAVAILABLE for "
                          f"target={list(target)}: {type(_e).__name__}",
                          flush=True)
                    _wr2 = None
                _added = 0
                if _wr2 is None:
                    _wr2 = {'actions': []}
                for (_J, _op, _dl) in _wr2['actions']:
                    _sd = tuple(_x + _d for _x, _d in zip(_J, _dl))
                    if (_op, _sd) in _seen_rows:
                        continue
                    _seen_rows.add((_op, _sd))
                    _ri = len(used_ibps)
                    used_ibps.append((_op, _sd))
                    _row = gm.apply_all_substitutions(
                        _strip(dict(gm.get_raw_equation(ibp_t, li_t, _op, _sd))),
                        subs)
                    cached_rows[_ri] = _row
                    for _k in _row:
                        occur.setdefault(_k, set()).add(_ri)
                    _added += 1
                _t_relap[0] += time.time() - _t_rl
                _cands = [idx for idx in sorted(occur.get(target, ()))
                          if idx not in used_set
                          and cached_rows[idx].get(target, 0) != 0]
                _n_full = len(_cands)
                _cands, _need_m2 = _cull(_cands, target, iteration)
                print(f"[relaporta] it={iteration} added={_added} rows -> "
                      f"{_n_full} cands, {len(_cands)} survive the cull",
                      flush=True)
            if not _cands:
                # DISTINCT from "no candidates at all": the library HAD a way
                # forward and the cull threw all of them away. This is the
                # failure the test is looking for, so it must not be silently
                # merged into no_action_found.
                success = False
                failure_reason = (f"cull_emptied: it={iteration} "
                                  f"M={_M_at(iteration)} alpha={_CULL_ALPHA} "
                                  f"full_cands={_n_full} relap={_n_relap[0]} "
                                  f"target={list(target)}")
                break
        n_cands = len(_cands)
        cand_counts.append(n_cands)
        if not _cands:
            found_action = None
            if _CFRAC:
                _inj = [i for i in range(len(used_ibps)) if i >= _n_truth_rows]
                _cover = [i for i in _inj if cached_rows.get(i, {}).get(target, 0) != 0]
                print(f"[inject-debug] no_action_found it={iteration} "
                      f"target={list(target)} active={_active(target)} "
                      f"injected_rows={len(_inj)} injected_covering_target="
                      f"{len(_cover)} of_those_unused="
                      f"{len([i for i in _cover if i not in used_set])} "
                      f"rejected_reintro={_n_reintro[0]} "
                      f"rejected_inactive={_n_inactive_intro[0]} "
                      f"total_introduced={_n_intro[0]}", flush=True)
        else:
            if _PICK == 'last':
                _sel = _cands[-1]
            elif _PICK == 'random':
                _sel = _rng.choice(_cands)
            elif _PICK == 'lex':
                # ON-POLICY LEX. ft_lex was built by taking the canon_corpus
                # recordings -- walked under minsumw -- and rewriting the answer
                # key to the lex-minimal action WITHOUT re-walking. Every label
                # was individually correct, but the states were only ever those
                # minsumw reaches. A model that then plays lex diverges from
                # that distribution immediately and further at every step, which
                # is what the depth-dependent rank collapse looks like:
                # ranks 1-8 for the first ~40 steps, hundreds to thousands after.
                #
                # This pick makes lex the ACTING rule, so states and labels come
                # from the same walk. Key is exactly build_ft_lex_data.py's:
                # min (ibp_op, delta), with the row index as a final tie-break
                # that never fires -- the rule was chosen because it gives a
                # unique pick 100% of the time.
                _sel = min(_cands,
                           key=lambda _ci: (
                               used_ibps[_ci][0],
                               tuple(used_ibps[_ci][1][_i] - target[_i]
                                     for _i in range(gm.N_INDICES)),
                               _ci))
            elif _PICK in _CANON:
                # CANONICAL pick: choose by a property of the substitution the
                # row produces, so the label is a deterministic function of the
                # state and is therefore learnable -- unlike "lowest closure row
                # index", which is an artifact of the truth engine's emission
                # order and carries no information the model can see.
                #
                # Cost is measured on the ACTIVE tail only: terms in a lower
                # sector, or below the start target in the total order, are
                # passengers that never need reducing, exactly as the worker
                # strips them. Counting them would rank rows by how much inert
                # baggage they carry.
                #
                # cached_rows are kept substitution-current by _apply_new_sub,
                # so the tail is already available -- no solve, no raw-equation
                # work. Ties break on (op, delta, idx) so the rule is total.
                _best = None
                for _ci in _cands:
                    _row = cached_rows[_ci]
                    _act = [k for k in _row
                            if k != target and sector_of(k) == S0
                            and not gm.is_master_for_sector(k, S0)
                            and tkey(k) <= kT]
                    if _PICK in ('minnew', 'minnewsumw'):
                        # OVERLAP WITH THE CURRENT EXPRESSION. minterms and
                        # minsumw charge for every active-tail integral the row
                        # introduces, whether or not that integral is ALREADY
                        # in the expression. One that is already there is not
                        # new work -- it merges into an existing term. So they
                        # systematically overcharge exactly the rows that reuse
                        # what is already on the table.
                        #
                        # The active tail is passenger-free by construction
                        # (_act filters to in-sector, non-master, at-or-above
                        # T), and expr is stripped the same way, so this
                        # compares like with like.
                        _new = [_k for _k in _act if _k not in expr]
                        _c = ((len(_new),) if _PICK == 'minnew' else
                              (sum(gm.weight(_k)[0] + gm.weight(_k)[1]
                                   for _k in _new),))
                    elif _PICK == 'minterms':
                        _c = (len(_act),)
                    elif _PICK == 'maxterms':
                        _c = (-len(_act),)
                    elif _PICK == 'minsumw':
                        _c = (sum(gm.weight(k)[0] + gm.weight(k)[1]
                                  for k in _act),)
                    else:  # minmaxw
                        _c = (max((gm.weight(k)[:2] for k in _act),
                                  default=(0, 0)),)
                    _op_t, _seed_t = used_ibps[_ci]
                    _key = (_c, _op_t,
                            tuple(_seed_t[i] - target[i]
                                  for i in range(gm.N_INDICES)), _ci)
                    if _best is None or _key < _best[0]:
                        _best = (_key, _ci, len(_act))
                _sel = _best[1]
                canon_cost.append(_best[2])
            else:
                _sel = _cands[0]
            found_action = (_sel,) + used_ibps[_sel]
        if found_action is None:
            success = False
            failure_reason = f"no_action_found: it={iteration}, target={list(target)}"
            break

        ph['search'] += time.time() - _tp
        idx, chosen_ibp_op, chosen_seed = found_action
        chosen_delta = tuple(chosen_seed[i] - target[i]
                             for i in range(gm.N_INDICES))
        # SAILIR_SKIP_ENUM: the enumeration is needed ONLY to write
        # `valid_actions` into the training record and to assert the chosen
        # action is enumerable. The REDUCTION itself never reads it -- the
        # action comes from the closure library and the solve is done on
        # cached_rows. Measured across 726 cull jobs: enum was 219,096s of
        # 237,872s = 92.1% of all compute, against 5,807s (2.4%) for the truth
        # replay. So for experiments that discard the recordings (the cull
        # sweeps), skipping it is a ~12x speedup for zero loss.
        #
        # It DOES remove a real safety check -- that the chosen action is one
        # the search would have been able to see. Keep it ON for any run whose
        # output is used as training data.
        if _SKIP_ENUM:
            valid_actions = None
        else:
            _tp = time.time()
            valid_actions = gm.enumerate_valid_actions(
                target, subs, ibp_t, li_t, shifts, S0, filter_lateral=False,
                enum_cache=enum_cache)
            ph['enum'] += time.time() - _tp
            if (chosen_ibp_op, chosen_delta) not in valid_actions:
                success = False
                failure_reason = f"action_not_valid: it={iteration}"
                break

        # Map every usable closure row to its position in valid_actions. One
        # dict build + |_cands| lookups per step -- no solving, no raw-equation
        # work, so this is cheap enough to leave on. A row whose action is not
        # in valid_actions is skipped rather than assumed present: only the
        # chosen row is guaranteed valid (checked above), the others are not.
        #
        # SKIP_ENUM: valid_actions is None, so there is nothing to map into and
        # no sample to emit. The walk itself reads neither.
        _label_idxs = []
        _label_ci = {}          # valid_actions position -> closure row index
        if not _SKIP_ENUM:
            _pos = {a: i for i, a in enumerate(valid_actions)}
            for _ci in _cands:
                _op_c, _seed_c = used_ibps[_ci]
                _d_c = tuple(_seed_c[i] - target[i]
                             for i in range(gm.N_INDICES))
                _p = _pos.get((_op_c, _d_c))
                if _p is not None:
                    _label_idxs.append(_p)
                    _label_ci[_p] = _ci
            _label_idxs.sort()
            n_labels_hist.append(len(_label_idxs))

        # ---- SAILIR_TRUTH_PICK=model -------------------------------------
        # MEASUREMENT, not a corpus generator: walk the model LIVE and find the
        # first step at which its own top pick is NOT one of the closure rows.
        #
        # Why this and not truth-depth. Truth depth asks whether the model
        # reproduces the single LEX row, which is one arbitrary tie-break among
        # ~17.9 equally-correct rows per step -- so it penalises a model for
        # picking a different, equally-reducing action. This asks the question
        # that matters: is the pick ONE OF the closure rows.
        #
        # It is fully ON-POLICY up to the miss, with no teacher forcing: every
        # step the model picks a library row, THAT row is the one consumed, so
        # the trajectory is the model's own. That is sound because any
        # consumption order of the library succeeds -- measured in this file's
        # own header (first/last/random reach success in 338/352/352 steps), so
        # staying inside the library keeps the walk valid no matter the order.
        # The moment it picks outside, the library no longer describes the
        # state and the measurement necessarily ends -- that step IS the datum.
        if _PICK == 'model':
            _pr = _model_probs(gm.filter_sector_only(expr, S0), subs,
                               valid_actions,
                               tuple(gm.get_sector_mask(S0)), target)
            _ord = list(np.argsort(-_pr))
            _top = int(_ord[0])
            _lab = set(_label_idxs)
            _anyrank = next((_r + 1 for _r, _v in enumerate(_ord)
                             if int(_v) in _lab), None)
            model_trace.append(dict(
                step=iteration, hit=bool(_top in _lab), anyrank=_anyrank,
                n_labels=len(_label_idxs), n_valid=len(valid_actions),
                top_prob=float(_pr[_top]),
                label_prob_sum=float(sum(_pr[i] for i in _label_idxs))))
            if _top not in _lab:
                success = False
                failure_reason = (f"model_left_library: it={iteration} "
                                  f"anyrank={_anyrank} "
                                  f"n_labels={len(_label_idxs)} "
                                  f"n_valid={len(valid_actions)}")
                break
            # Consume the row the MODEL chose, not the provisional one.
            idx = _label_ci[_top]
            chosen_ibp_op, chosen_seed = used_ibps[idx]
            chosen_delta = tuple(chosen_seed[_i] - target[_i]
                                 for _i in range(gm.N_INDICES))
        # ------------------------------------------------------------------

        # v6 MULTILABEL equivalence mask REMOVED 2026-08-04. It ran a full
        # raw-equation + substitution + solve for EVERY valid action at every
        # step — measured 4.4s of a 23.1s profile (19% of total runtime) — and
        # returned nothing: avg_equiv came out 1.0 (no action ever shared a
        # child expression with the chosen one). It was added on the strength
        # of an "equivalent actions" finding that was itself an artifact of a
        # missing SAILIR_TOPOLOGY (every solve returned None, so None==None
        # compared equal). Do NOT reinstate without first measuring avg_equiv
        # > 1.0 under a topology-guarded positive control.
        if (os.environ.get('SAILIR_MAX_SAMPLES')
                and len(samples) >= int(os.environ['SAILIR_MAX_SAMPLES'])):
            success = False
            failure_reason = 'sample_cap_for_profiling'
            break
        # SKIP_ENUM: no enumeration means no valid_actions and no label
        # indices, so there is no well-formed sample. The WALK is
        # unaffected -- the solve and substitution below still run.
        if True:   # record the trajectory even with enum off
            samples.append({
                'scramble_id': 0,
                'source': 'truth_onestep_v2',
                'sector_id': S0,
                'sector_mask': list(gm.get_sector_mask(S0)),
                'n_sym_ops': 0,
                'n_base_ops': len(shifts),
                'step': iteration,
                'target': list(target),
                'target_weight': list(gm.weight(target)[:2]),
                'expr': gm.expr_to_json(gm.filter_sector_only(expr, S0)),
                'subs': gm.subs_to_json(subs),
                **({} if _SKIP_ENUM else {
                    'valid_actions': [[o, list(d)] for o, d in valid_actions],
                    'num_valid_actions': len(valid_actions)}),
                'chosen_action': [chosen_ibp_op, list(chosen_delta)],
                **({} if _SKIP_ENUM else {
                    'chosen_action_idx':
                        valid_actions.index((chosen_ibp_op, chosen_delta))}),
                # MULTI-LABEL: every unused closure row that solves this target is
                # an equally correct action -- measured, not assumed: picking the
                # first, last, or a random one all reach success=True (338 / 352 /
                # 352 steps on this integral). There are typically 25-45 of them,
                # so scoring the model against the single 'chosen_action' measures
                # imitation of the lowest-row-index convention rather than whether
                # it picked a reducing action. Indices into valid_actions, so
                # downstream code needs no action matching.
                **({} if _SKIP_ENUM else {
                    'valid_label_idxs': _label_idxs,
                    'n_valid_labels': len(_label_idxs)}),
            })

        if os.environ.get('SAILIR_STEP_LOG', '0') == '1':
            _now = time.time()
            print(f"[step {iteration}/{len(used_ibps)}] "
                  f"dt={_now - globals().get('_last_step_t', _now):.2f}s "
                  f"subs={len(subs)} n_valid={len(valid_actions)} "
                  f"enum_cum={ph['enum']:.1f}s", flush=True)
            globals()['_last_step_t'] = _now
        _tp = time.time()
        # Under strip mode the row is already active-only, so sol is too; the
        # explicit _strip on expr catches nothing new but costs one pass and
        # documents the invariant. Rows stay stripped because _apply_new_sub
        # only ever inserts keys drawn from sol.
        sol = gm.solve_ibp_for(dict(cached_rows[idx]), target)
        used_set.add(idx)
        cached_rows.pop(idx, None)
        subs[target] = sol
        _sub_step.setdefault(target, iteration)
        _apply_new_sub(target, sol)
        if enum_cache is not None:
            enum_cache.add_sub(target, sol)
        expr = _strip(gm.apply_substitution(expr, target, sol))
        ph['apply'] += time.time() - _tp
    else:
        success = False
        failure_reason = "max_iterations_reached"

    if success:
        with open(args.output, 'w') as f:
            for s in samples:
                f.write(json.dumps(s) + '\n')
    if _PICK == 'model':
        # The datum is the FIRST MISS: how many consecutive steps the model
        # acted entirely on its own and stayed inside the closure library.
        # 'survived' == len(model_trace) when it missed, and equals the full
        # walk length if it never did (walk completed inside the library).
        _hits = [t for t in model_trace if t['hit']]
        _miss = next((t for t in model_trace if not t['hit']), None)
        _out = args.output + '.modeltrace.json'
        with open(_out, 'w') as _f:
            json.dump(dict(integral=list(T), ckpt=os.environ.get('SAILIR_MODEL_CKPT'),
                           survived=len(_hits), n_steps=len(model_trace),
                           closure_len=len(used_ibps),
                           left_library=_miss is not None,
                           miss=_miss, trace=model_trace), _f)
        _an = [t['anyrank'] for t in model_trace if t['anyrank']]
        print(f"MODELWALK survived={len(_hits)} of closure={len(used_ibps)} "
              f"left_library={_miss is not None} "
              f"median_anyrank={sorted(_an)[len(_an)//2] if _an else 'na'} "
              f"mean_label_mass={sum(t['label_prob_sum'] for t in model_trace)/max(len(model_trace),1):.4f} "
              f"-> {_out}", flush=True)

    if cand_counts:
        _c = sorted(cand_counts)
        _n = len(_c)
        print(f"CANDS pick={_PICK} steps={_n} min={_c[0]} "
              f"p25={_c[_n//4]} median={_c[_n//2]} p75={_c[3*_n//4]} max={_c[-1]} "
              f"n_with_1_choice={sum(1 for x in _c if x == 1)} "
              f"mean={sum(_c)/_n:.1f}", flush=True)
    if canon_cost:
        _K = sorted(canon_cost); _n = len(_K)
        print(f"CANON rule={_PICK} steps={_n} active_tail_terms: min={_K[0]} "
              f"median={_K[_n//2]} max={_K[-1]} mean={sum(_K)/_n:.1f}", flush=True)
    if n_labels_hist:
        _L = sorted(n_labels_hist); _n = len(_L)
        print(f"LABELS steps={_n} min={_L[0]} p25={_L[_n//4]} median={_L[_n//2]} "
              f"p75={_L[3*_n//4]} max={_L[-1]} mean={sum(_L)/_n:.1f} "
              f"n_single={sum(1 for x in _L if x == 1)}", flush=True)
    # Realized corruption, reported so the CONFIGURED fraction is never
    # confused with what actually landed: the guard can reject a draw (it
    # re-introduces an already-solved integral, or reaches outside the active
    # set) and fall through its retries, silently lowering the effective rate.
    _corrupt_stats = (
        f"injected={n_injected} events_ok={_n_ok[0]} "
        f"attempts={_n_attempt[0]} rej_reintro={_n_reintro[0]} "
        f"rej_inactive={_n_inactive_intro[0]} "
        f"skipped_events={_n_skipped[0]} noop={_n_noop[0]} "
        # injected rows never consumed at close. The loop ends when the active
        # bucket empties, so leftovers are harmless -- but they explain any gap
        # between identities injected and extra steps taken, which should
        # otherwise be 1:1.
        f"injected_unused={sum(1 for _i in range(_n_truth_rows, len(used_ibps)) if _i not in used_set)} "
        f"truth_rows_unused={sum(1 for _i in range(_n_truth_rows) if _i not in used_set)} "
        f"depth_accepted={dict(sorted(_ns_ok.items()))} "
        f"depth_rejected={dict(sorted(_ns_rej.items()))} "
        f"identities_per_event={dict(sorted(_ns_used.items()))}"
        if _CFRAC else "")
    # The smallest M that would have sufficed at each step of THIS walk. The
    # max is the requirement for the trajectory this rule actually took -- which
    # is a different trajectory per pick rule, so it must be measured per rule
    # rather than carried over from the lex-walked recordings.
    if _CULL_M >= 0:
        _nv = sorted(k for k, v in _need_hist.items() for _ in range(v))
        _nq = ""
        if _nv:
            _nq = (f"need_M_max={_nv[-1]} "
                   f"need_M_p50={_nv[len(_nv) // 2]} "
                   f"need_M_p99={_nv[int(0.99 * len(_nv))]} "
                   f"need_M_steps={len(_nv)} ")
        print(f"CULLSTATS cull_M={_CULL_M} cull_alpha={_CULL_ALPHA} "
              f"cull_kept={_cull_kept[0]} cull_full={_cull_full[0]} "
              f"cull_frac={_cull_kept[0] / max(_cull_full[0], 1):.4f} "
              f"cull_lost_lex_pick={_cull_lost_lex[0]} {_nq}"
              f"relaporta={_n_relap[0]} "
              f"relaporta_unavailable={_n_relap_fail[0]} "
              f"direct_rescue={_n_direct_rescue[0]} "
              f"relaporta_time={_t_relap[0]:.1f}s",
              flush=True)
    print(f"STATS target={','.join(map(str, T))} sector={S0} "
          f"L={bin(S0).count('1')} dots={r0 - bin(S0).count('1')} "
          f"closure={len(used_ibps)} success={success} "
          f"samples={len(samples) if success else 0} "
          f"{_corrupt_stats} "
          f"reason={failure_reason} truth_time={t_truth:.1f}s "
          f"search={ph['search']:.1f}s enum={ph['enum']:.1f}s "
          f"apply={ph['apply']:.1f}s raw_gen={raw_stat['time']:.1f}s"
          f"/{raw_stat['calls']}calls "
          f"total_time={time.time()-t0:.1f}s "
          f"enumcache={enum_cache.stats() if enum_cache else 'off'}", flush=True)


if __name__ == '__main__':
    main()
