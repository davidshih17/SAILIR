#!/usr/bin/env python
"""TRUTH -> TRAINING DATA (2026-07-31): run a full truth-engine reduction and
emit each in-sector step as a training sample in the EXACT schema of
data-gen/generate_multisector_data.py (sector_id, expr, subs, valid_actions,
chosen_action, chosen_action_idx).

Per step of TruthEngine.reduce() with target J in the START target's sector:
  - expr    = sector-local view (filter_sector_only), as in the generator
  - subs    = accumulated in-sector eliminations {target: tail}
  - valid_actions = generator's enumerate_valid_actions (same filters as
    production data-gen: filter_higher on, filter_lateral off)
  - chosen  = the truth action (op, delta); if it is NOT in the enumerated
    list the sample is DROPPED and counted (action_not_valid) — the drop
    rate is a key feasibility number.
Cross-sector (subsector-debris) steps are skipped and counted.

Usage:
  SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1 truth_record_training.py \
      --integral 1,1,2,0,1,... --output out.jsonl [--dr 1 --ds 1]
      [--max-record-steps 2000]
Prints a single STATS line at the end (parsed by the sizing sweep).
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

P = 1009


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--integral', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--dr', type=int, default=1)
    ap.add_argument('--ds', type=int, default=1)
    ap.add_argument('--max-record-steps', type=int, default=2000,
                    help='stop RECORDING after this many in-sector steps '
                         '(the reduction itself continues); bounds the '
                         'quadratic enumerate-over-subs cost')
    ap.add_argument('--trivialsector', default=os.path.join(
        _tc.TOPO_DIR, "kira_validate/sectormappings/GR/trivialsector"))
    ap.add_argument('--one-step-only', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='DEFAULT ON: stop the reduction once no start-sector '
                         'non-master remains (the worker contract: bucket '
                         'drained = one weight-level reduction) — skips the '
                         'subsector tail. --no-one-step-only for a full '
                         'reduction to masters')
    args = ap.parse_args()

    T = tuple(int(x) for x in args.integral.strip("'\"").split(","))
    S0 = sector_of(T)
    r0, s0 = rs_of(T)

    # data-gen module init: SAME topology, prime, templates as production gen
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

    eng = TruthEngine(_tc.TOPO_DIR, args.trivialsector)

    subs = {}
    stats = dict(recorded=0, action_not_valid=0, cross_sector=0,
                 record_capped=0, deps_added=0)
    out = open(args.output, 'w')
    t_enum = [0.0]

    def materialize_deps(J, rules):
        """Add the dependency closure of J's rule to `subs`, dependencies
        first (the Laporta build exposes a rule's pivot only after its deps
        are substituted; the worker mirrors this with its solution store —
        see TruthEngine.worker_replay). Stored tails are already fully
        reduced, so subs[dep] = tail directly."""
        added = 0
        stack = [(J, False)]
        seen = set()
        while stack:
            K, expanded = stack.pop()
            if K in seen or (K in subs and K != J):
                continue
            if expanded:
                seen.add(K)
                if K != J and K not in subs:
                    subs[K] = dict(rules[K][0])
                    added += 1
                continue
            stack.append((K, True))
            for D in rules[K][3]:
                if D not in subs and D not in seen and D in rules:
                    stack.append((D, False))
        return added

    class _StartSectorDone(Exception):
        pass

    def recorder(step, J, op, seed, tail, expr, rules):
        if sector_of(J) != S0:
            stats['cross_sector'] += 1
            if args.one_step_only and not any(
                    sector_of(K) == S0 for K in expr if K != J):
                raise _StartSectorDone
            return
        if stats['recorded'] >= args.max_record_steps:
            stats['record_capped'] += 1
            return
        stats['deps_added'] += materialize_deps(J, rules)
        delta = tuple(a - b for a, b in zip(seed, J))
        t0 = time.time()
        valid = gm.enumerate_valid_actions(J, subs, ibp_t, li_t, shifts, S0,
                                           filter_higher=True,
                                           filter_lateral=False)
        t_enum[0] += time.time() - t0
        if (op, delta) not in valid:
            stats['action_not_valid'] += 1
            subs[J] = dict(tail)      # keep the store faithful regardless
            return
        sample = {
            'scramble_id': 0,
            'source': 'truth_engine',
            'sector_id': S0,
            'sector_mask': list(gm.get_sector_mask(S0)),
            'n_sym_ops': 0,
            'n_base_ops': len(shifts),
            'step': step,
            'target': list(J),
            'target_weight': list(gm.weight(J)[:2]),
            'expr': gm.expr_to_json(gm.filter_sector_only(expr, S0)),
            'subs': gm.subs_to_json(subs),
            'valid_actions': [[o, list(d)] for o, d in valid],
            'num_valid_actions': len(valid),
            'chosen_action': [op, list(delta)],
            'chosen_action_idx': valid.index((op, delta)),
        }
        out.write(json.dumps(sample) + '\n')
        stats['recorded'] += 1
        subs[J] = dict(tail)
        if args.one_step_only and not any(
                sector_of(K) == S0 and not gm.is_master_for_sector(K, S0)
                for K in expr if K != J):
            raise _StartSectorDone

    t0 = time.time()
    try:
        res = eng.reduce(T, dr=args.dr, ds=args.ds, recorder=recorder)
    except _StartSectorDone:
        res = dict(steps=-1, final_expr={})
        print("one-step-only: start sector drained — stopping early",
              flush=True)
    out.close()
    el = time.time() - t0
    print(f"STATS target={','.join(map(str, T))} sector={S0} L={bin(S0).count('1')} "
          f"rs=({r0},{s0}) dots={r0 - bin(S0).count('1')} "
          f"steps={res['steps']} recorded={stats['recorded']} "
          f"not_valid={stats['action_not_valid']} "
          f"cross_sector={stats['cross_sector']} "
          f"capped={stats['record_capped']} "
          f"deps_added={stats['deps_added']} "
          f"masters={len(res['final_expr'])} "
          f"time={el:.1f}s enum_time={t_enum[0]:.1f}s", flush=True)


if __name__ == '__main__':
    main()
