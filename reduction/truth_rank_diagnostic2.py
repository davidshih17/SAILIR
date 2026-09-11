#!/usr/bin/env python
"""MODEL-RANK DIAGNOSTIC v2 — on the WORKER-DYNAMICS truth trajectory
(TruthEngine.reduce: at every step target the highest total-lex-weight
non-master of the current expression, exactly the worker's rule).

For the two 16h-stuck g885 integrals, inline reduce()'s loop and at every
step, BEFORE applying the pivot's rule, measure:
  in_valid  enumerate_valid_actions offers the truth (op,delta) at the target
  rank      model rank of the truth action among the candidates
  in_top20  makes the beam intake (top_k = beam_width//2 = 20)
Stops when the ONE-STEP success criterion fires (no non-master at-or-above
the start in the total order) — the worker's actual goal.

Writes results/truth/rank_diag2.pkl and a per-step table to stdout.
"""
import os, sys, pickle, time
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
import numpy as np
import torch
torch.set_num_threads(4)

from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc

P = 1009
ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
ibp_env.set_prime(P)
ibp_env.set_paper_masters_only(True)
from canonical_masters import apply_canonical_masters
apply_canonical_masters()
from sailir.ibp_env import enumerate_valid_actions, is_master
from beam_search_utils import get_sector_mask

from truth_engine import TruthEngine, sector_of, rs_of

CKPT = os.path.join(ROOT, "checkpoints/gravity3L_canon10x_nosubs/best_model.pt")
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
from sailir.classifier_nosubs import IBPActionClassifierNoSubs
topo = Topology.from_dir(_tc.TOPO_DIR)
model = IBPActionClassifierNoSubs(
    prime=(ck.get('args') or {}).get('prime', P),
    n_indices=topo.n_indices, n_denominators=topo.n_denominators,
    n_ibp_ops=topo.n_actions)
model.load_state_dict(ck['model_state_dict'])
model.eval()

sys.argv += ['--v7-cpus', '4']    # beam_search_v7 import-time CPU pin reads argv
import beam_search_v7 as bs7

triv = os.path.join(_tc.TOPO_DIR, "kira_validate/sectormappings/GR/trivialsector")
eng = TruthEngine(_tc.TOPO_DIR, triv)
K_BEAM = 20

STUCK = [
    (2, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0),
    (1, 0, 1, 0, 1, 1, -2, 0, 1, 1, 0, 0, 0, 0, 0),
]

out = {}
for T in STUCK:
    kT = eng.tkey(T)
    tsec_mask = get_sector_mask(T)
    expr = {T: 1 % P}
    rs_prefix = {}
    caps = {}
    rows = []
    print(f"\n===== {list(T)}", flush=True)
    for step in range(5000):
        nm = [J for J in expr if not is_master(J) and sector_of(J) != 0
              and sector_of(J) not in eng.trivial]
        for J in [j for j in expr if sector_of(j) != 0
                  and sector_of(j) in eng.trivial]:
            expr.pop(J)
        # ONE-STEP success: nothing at-or-above the start remains
        active = [J for J in nm if eng.tkey(J) <= kT]
        if not active:
            break
        J = min(nm, key=eng.tkey)
        S = sector_of(J)
        r, s = rs_of(J)
        want = (max(r + 1, caps.get(S, (0, 0))[0]),
                max(s + 1, caps.get(S, (0, 0))[1]))
        rules = eng.build_system(S, *want, verbose=False)
        caps[S] = want
        if J not in rules:
            # reduce()'s escalation: targeted local search, then adaptive cap
            # escalation while the projected system stays cheap
            ok = eng.build_rule_for(J, rules, verbose=False)
            tries = 0
            while not ok and tries < 3:
                nxt = (want[0] + 1, want[1] + 1)
                if sum(1 for _ in eng._seeds(S, *nxt)) > 25000:
                    break
                rules = eng.build_system(S, *nxt, verbose=False)
                caps[S] = want = nxt
                ok = J in rules or eng.build_rule_for(J, rules, verbose=False)
                tries += 1
            if not ok and J not in rules:
                raise RuntimeError(f"no rule for {list(J)} at caps {want}")
        tail, op, seed = rules[J][:3]
        delta = tuple(x - y for x, y in zip(seed, J))
        # ---- diagnostic at this genuine worker decision point ----
        valid = enumerate_valid_actions(J, rs_prefix, eng.env.ibp_t,
                                        eng.env.li_t, eng.env.shifts,
                                        'subsector')
        try:
            vidx = valid.index((op, delta))
        except ValueError:
            vidx = -1
        rank = p_t = p_top = None
        if vidx >= 0:
            bd = [(expr, rs_prefix, valid, tsec_mask, J)]
            b = bs7.prepare_batched_input_v5_dummy(
                bd, 'cpu', max_actions=max(len(valid), 1))
            with torch.no_grad():
                _, probs = model(
                    b['expr_integrals'], b['expr_coeffs'], b['expr_mask'],
                    b['sub_keys'], b['sub_repl_ints'], b['sub_repl_coeffs'],
                    b['sub_repl_mask'], b['sub_mask'],
                    b['action_ibp_ops'], b['action_deltas'], b['action_mask'],
                    b['sector_mask'], b['target_integral'],
                )
            row = probs[0, :len(valid)].numpy()
            order = np.argsort(-row)
            rank = int(np.where(order == vidx)[0][0]) + 1
            p_t, p_top = float(row[vidx]), float(row[order[0]])
        rows.append(dict(step=step, J=J, op=op, delta=delta,
                         n_valid=len(valid), in_valid=vidx >= 0, rank=rank,
                         p_truth=p_t, p_top=p_top, expr_size=len(expr),
                         n_active=len(active)))
        print(f"  step {step:3d} J={list(J)} valid={len(valid)} "
              f"in_valid={vidx >= 0} rank={rank} "
              f"p_truth={None if p_t is None else round(p_t, 4)} "
              f"p_top={None if p_top is None else round(p_top, 4)} "
              f"expr={len(expr)} active={len(active)}", flush=True)
        # ---- apply the rule (worker net effect) ----
        co = expr.pop(J)
        for Kk, cK in tail.items():
            v = (expr.get(Kk, 0) + co * cK) % P
            if v:
                expr[Kk] = v
            else:
                expr.pop(Kk, None)
        rs_prefix[J] = dict(tail)
    n = len(rows)
    ranks = [x["rank"] for x in rows if x["rank"] is not None]
    print(f"\n  SUMMARY {list(T)}: {n} worker steps to one-step success")
    print(f"  in enumeration: {sum(1 for x in rows if x['in_valid'])}/{n}")
    if ranks:
        print(f"  rank: median={int(np.median(ranks))} mean={np.mean(ranks):.0f} "
              f"max={max(ranks)}  in-top-{K_BEAM}: "
              f"{sum(1 for x in ranks if x <= K_BEAM)}/{len(ranks)}")
    out[T] = rows
    eng.systems.clear()

with open(os.path.join(ROOT, "results/truth/rank_diag2.pkl"), "wb") as f:
    pickle.dump(out, f)
print("\nsaved -> results/truth/rank_diag2.pkl")
