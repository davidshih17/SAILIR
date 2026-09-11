#!/usr/bin/env python
"""TRUTH -> TRAINING DATA v2 (2026-08-01): DATA-GEN-FAITHFUL recording.

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
import sys
import json
import time
import argparse

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))
sys.path.insert(0, os.path.join(ROOT, "data-gen"))

from sailir.topology import Topology
import topo_config as _tc
import generate_multisector_data as gm
from truth_engine import TruthEngine, sector_of, rs_of
from symmetry_route import tkey

P = 1009


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--integral', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--dr', type=int, default=1)
    ap.add_argument('--ds', type=int, default=1)
    args = ap.parse_args()

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
    eng = TruthEngine(_tc.TOPO_DIR, os.path.join(
        _tc.TOPO_DIR, "kira_validate/sectormappings/GR/trivialsector"))
    wr = eng.worker_replay(T, dr=args.dr, ds=args.ds, verbose=True)
    # the truth closure as a scramble-style action record + the exact result
    used_ibps = [(op, tuple(x + d for x, d in zip(J, delta)))
                 for (J, op, delta) in wr['actions']]
    truth_result = {k: v % P for k, v in wr['result_expr'].items()}
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
    expr = {T: 1}
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
    cached_rows = {}
    occur = {}                    # integral -> set of row idxs containing it
    for idx, (ibp_op, seed) in enumerate(used_ibps):
        row = dict(gm.get_raw_equation(ibp_t, li_t, ibp_op, seed))
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
    for iteration in range(len(used_ibps) + 500):
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

            if _nf(final) != _nf(truth_result):
                success = False
                failure_reason = (f"end_state_mismatch_normalform: "
                                  f"{len(final)} vs {len(truth_result)} terms")
            break

        target = max(bucket.keys(), key=gm.weight)

        _tp = time.time()
        found_action = None
        for idx in sorted(occur.get(target, ())):
            if idx in used_set:
                continue
            if cached_rows[idx].get(target, 0) != 0:
                found_action = (idx,) + used_ibps[idx]
                break
        if found_action is None:
            success = False
            failure_reason = f"no_action_found: it={iteration}, target={list(target)}"
            break

        ph['search'] += time.time() - _tp
        idx, chosen_ibp_op, chosen_seed = found_action
        chosen_delta = tuple(chosen_seed[i] - target[i]
                             for i in range(gm.N_INDICES))
        _tp = time.time()
        valid_actions = gm.enumerate_valid_actions(
            target, subs, ibp_t, li_t, shifts, S0, filter_lateral=False,
            enum_cache=enum_cache)
        ph['enum'] += time.time() - _tp
        if (chosen_ibp_op, chosen_delta) not in valid_actions:
            success = False
            failure_reason = f"action_not_valid: it={iteration}"
            break

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
            'valid_actions': [[o, list(d)] for o, d in valid_actions],
            'num_valid_actions': len(valid_actions),
            'chosen_action': [chosen_ibp_op, list(chosen_delta)],
            'chosen_action_idx':
                valid_actions.index((chosen_ibp_op, chosen_delta)),
        })

        if os.environ.get('SAILIR_STEP_LOG', '0') == '1':
            _now = time.time()
            print(f"[step {iteration}/{len(used_ibps)}] "
                  f"dt={_now - globals().get('_last_step_t', _now):.2f}s "
                  f"subs={len(subs)} n_valid={len(valid_actions)} "
                  f"enum_cum={ph['enum']:.1f}s", flush=True)
            globals()['_last_step_t'] = _now
        _tp = time.time()
        sol = gm.solve_ibp_for(dict(cached_rows[idx]), target)
        used_set.add(idx)
        cached_rows.pop(idx, None)
        subs[target] = sol
        _apply_new_sub(target, sol)
        if enum_cache is not None:
            enum_cache.add_sub(target, sol)
        expr = gm.apply_substitution(expr, target, sol)
        ph['apply'] += time.time() - _tp
    else:
        success = False
        failure_reason = "max_iterations_reached"

    if success:
        with open(args.output, 'w') as f:
            for s in samples:
                f.write(json.dumps(s) + '\n')
    print(f"STATS target={','.join(map(str, T))} sector={S0} "
          f"L={bin(S0).count('1')} dots={r0 - bin(S0).count('1')} "
          f"closure={len(used_ibps)} success={success} "
          f"samples={len(samples) if success else 0} "
          f"reason={failure_reason} truth_time={t_truth:.1f}s "
          f"search={ph['search']:.1f}s enum={ph['enum']:.1f}s "
          f"apply={ph['apply']:.1f}s raw_gen={raw_stat['time']:.1f}s"
          f"/{raw_stat['calls']}calls "
          f"total_time={time.time()-t0:.1f}s "
          f"enumcache={enum_cache.stats() if enum_cache else 'off'}", flush=True)


if __name__ == '__main__':
    main()
