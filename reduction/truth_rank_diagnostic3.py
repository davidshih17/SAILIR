#!/usr/bin/env python
"""MODEL-RANK DIAGNOSTIC v3 — genuine worker dynamics.

Worker-form truth trajectory: state = (expr, rs store). Each step targets the
HIGHEST total-lex-weight active non-master J (the worker's rule) and plays a
truth-system closure equation whose RAW form, resolved by the CURRENT rs
store, contains J (preferring J's own generator). Worker semantics exactly:
solve for J, substitute into expr (introducing the equation's other resolved
terms), add rs[J]. Each closure pivot is solved at most once -> terminates.
At every step, BEFORE playing, the model ranks all enumerated valid actions;
we record the truth action's rank.

Writes results/truth/rank_diag3.pkl + per-step table.
"""
import os, sys, pickle
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
from sailir.ibp_env import (enumerate_valid_actions, is_master,
                            get_raw_equation)
from beam_search_utils import get_sector_mask
from truth_engine import TruthEngine, sector_of

sys.argv += ['--v7-cpus', '4']    # beam_search_v7 import-time CPU pin
import beam_search_v7 as bs7

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

triv = os.path.join(_tc.TOPO_DIR, "kira_validate/sectormappings/GR/trivialsector")
eng = TruthEngine(_tc.TOPO_DIR, triv)
K_BEAM = 20

with open(os.path.join(ROOT, "results/truth/g885_stuck2_onestep.pkl"), "rb") as f:
    closures = pickle.load(f)


def resolve(raw, rs):
    """Worker resolution: substitute every rs entry present in the equation."""
    cached = dict(raw)
    while True:
        hit = next((K for K in cached if K in rs), None)
        if hit is None:
            return cached
        co = cached.pop(hit)
        for K, cK in rs[hit].items():
            v = (cached.get(K, 0) + co * cK) % P
            if v:
                cached[K] = v
            else:
                cached.pop(K, None)


out = {}
for T, r in closures.items():
    if r is None:
        continue
    # closure generator equations: pivot -> (op, seed); plus the full list for
    # fallback lookup of ANY equation containing the current target
    gens = {J: (op, tuple(x + y for x, y in zip(J, shift)))
            for J, op, shift in r["actions"]}
    all_eqs = [(J, op, seed) for J, (op, seed) in gens.items()]
    kT = eng.tkey(T)
    tsec_mask = get_sector_mask(T)
    expr = {T: 1 % P}
    rs = {}
    rows = []
    print(f"\n===== {list(T)}  closure size {len(gens)}", flush=True)
    for step in range(3 * len(gens) + 10):
        for J in [j for j in list(expr) if sector_of(j) != 0
                  and sector_of(j) in eng.trivial]:
            expr.pop(J)
        nm = [J for J in expr if expr[J] % P and not is_master(J)
              and sector_of(J) != 0]
        active = [J for J in nm if eng.tkey(J) <= kT]
        if not active:
            print(f"  DRAINED at step {step}", flush=True)
            break
        J = min(nm, key=eng.tkey)
        # pick the truth action: J's own generator if worker-valid now, else
        # any closure equation whose rs-resolved raw contains J
        pick = None
        cand_list = ([(J, *gens[J])] if J in gens else []) + \
                    [(p, op, seed) for p, op, seed in all_eqs if p != J]
        for p, op, seed in cand_list:
            raw = get_raw_equation(eng.env.ibp_t, eng.env.li_t, op, seed)
            res = resolve(raw, rs)
            if J in res and res[J] % P:
                pick = (op, seed, res)
                break
        if pick is None:
            raise RuntimeError(f"no closure equation eliminates {list(J)} "
                               f"at this state — closure insufficient for "
                               f"worker-dynamics order")
        op, seed, res = pick
        delta = tuple(x - y for x, y in zip(seed, J))
        # ---- diagnose ----
        valid = enumerate_valid_actions(J, rs, eng.env.ibp_t, eng.env.li_t,
                                        eng.env.shifts, 'subsector')
        try:
            vidx = valid.index((op, delta))
        except ValueError:
            vidx = -1
        rank = p_t = p_top = None
        if vidx >= 0 and valid:
            bd = [(expr, rs, valid, tsec_mask, J)]
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
            rowp = probs[0, :len(valid)].numpy()
            order = np.argsort(-rowp)
            rank = int(np.where(order == vidx)[0][0]) + 1
            p_t, p_top = float(rowp[vidx]), float(rowp[order[0]])
        rows.append(dict(step=step, J=J, op=op, delta=delta,
                         n_valid=len(valid), in_valid=vidx >= 0, rank=rank,
                         p_truth=p_t, p_top=p_top, expr_size=len(expr),
                         n_active=len(active)))
        print(f"  step {step:3d} J={list(J)} valid={len(valid)} "
              f"in_valid={vidx >= 0} rank={rank} "
              f"p_truth={None if p_t is None else round(p_t, 4)} "
              f"p_top={None if p_top is None else round(p_top, 4)} "
              f"expr={len(expr)} active={len(active)}", flush=True)
        # ---- play (worker semantics) ----
        co = expr.pop(J)
        inv = pow(res[J], P - 2, P)
        tail = {K: (-c * inv) % P for K, c in res.items()
                if K != J and c % P}
        rs[J] = tail
        for Kk, cK in tail.items():
            v = (expr.get(Kk, 0) + co * cK) % P
            if v:
                expr[Kk] = v
            else:
                expr.pop(Kk, None)
        # keep the store consistent: fold the new solve into older rs tails
        for Kp in list(rs):
            if Kp == J:
                continue
            t = rs[Kp]
            if J in t:
                cJ = t.pop(J)
                for Kk, cK in tail.items():
                    v = (t.get(Kk, 0) + cJ * cK) % P
                    if v:
                        t[Kk] = v
                    else:
                        t.pop(Kk, None)
    n = len(rows)
    ranks = [x["rank"] for x in rows if x["rank"] is not None]
    print(f"\n  SUMMARY {list(T)}: {n} worker steps")
    print(f"  truth action in enumeration: "
          f"{sum(1 for x in rows if x['in_valid'])}/{n}")
    if ranks:
        print(f"  rank: median={int(np.median(ranks))} mean={np.mean(ranks):.0f} "
              f"max={max(ranks)}  in-top-{K_BEAM}: "
              f"{sum(1 for x in ranks if x <= K_BEAM)}/{len(ranks)}")
    out[T] = rows
with open(os.path.join(ROOT, "results/truth/rank_diag3.pkl"), "wb") as f:
    pickle.dump(out, f)
print("\nsaved -> results/truth/rank_diag3.pkl")
