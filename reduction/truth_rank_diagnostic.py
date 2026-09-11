#!/usr/bin/env python
"""MODEL-RANK DIAGNOSTIC on the certified truth one-step paths of the two
16h-stuck g885 integrals (sector 821).

Walk the truth action sequence keeping the worker-equivalent state
(expr, resolved-subs prefix). At every PLAYABLE step (truth pivot J present in
the expression) determine:
  tied?       is J among the worker's targets (max-(r,s) non-masters)
  in_valid?   does enumerate_valid_actions offer the truth (op,delta) for J
  rank        model rank of the truth action among the valid list
  in_top20    would it enter the beam (K = beam_width//2 = 20)
Aggregates at the end tell whether the failure is enumeration coverage,
target choice, or model ranking — and how far off the ranking is.

Writes results/truth/rank_diag.pkl + prints a per-step table.
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
from sailir.ibp_env import (get_raw_equation, enumerate_valid_actions,
                            weight, is_master)
from beam_search_utils import get_sector_mask

env = ibp_env.IBPEnvironment()

# model: exactly as onestep_worker_v7 constructs the nosubs variant
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

import beam_search_v7 as bs7

with open(os.path.join(ROOT, "results/truth/g885_stuck2_onestep.pkl"), "rb") as f:
    data = pickle.load(f)

K_BEAM = 20      # top_k = beam_width//2 with beam_width=40
out = {}
for T, r in data.items():
    if r is None:
        continue
    acts, sol = r["actions"], r["sol_store"]
    tsec_mask = get_sector_mask(T)
    expr = {T: 1}
    rs_prefix = {}       # ordered: solved pivots so far (worker RS analogue)
    rows = []
    for i, (J, op, shift) in enumerate(acts):
        delta = tuple(shift)
        playable = J in expr
        if playable:
            nm = [k for k, v in expr.items() if v % P and not is_master(k)]
            mw = max((weight(k)[0], weight(k)[1]) for k in nm) if nm else None
            tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
            in_tied = J in tied
            valid = enumerate_valid_actions(J, rs_prefix, env.ibp_t, env.li_t,
                                            env.shifts, 'subsector')
            try:
                vidx = valid.index((op, delta))
            except ValueError:
                vidx = -1
            rank = prob_t = prob_top = None
            if vidx >= 0:
                bd = [(expr, rs_prefix, valid, tsec_mask, J)]
                b = bs7.prepare_batched_input_v5_dummy(bd, 'cpu',
                                                       max_actions=max(len(valid), 1))
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
                prob_t, prob_top = float(row[vidx]), float(row[order[0]])
            rows.append(dict(i=i, J=J, op=op, delta=delta, n_valid=len(valid),
                             in_tied=in_tied, n_tied=len(tied), in_valid=vidx >= 0,
                             rank=rank, prob_truth=prob_t, prob_top=prob_top))
            print(f"  step {i:3d} J={list(J)} tied={in_tied}({len(tied)}) "
                  f"valid={len(valid)} in_valid={vidx >= 0} rank={rank} "
                  f"p_truth={prob_t if prob_t is None else round(prob_t, 4)} "
                  f"p_top={prob_top if prob_top is None else round(prob_top, 4)}",
                  flush=True)
        # advance state: apply solve if pivot present; always record RS entry
        if J in expr:
            co = expr.pop(J)
            for Kk, cK in sol[J].items():
                v = (expr.get(Kk, 0) + co * cK) % P
                if v:
                    expr[Kk] = v
                else:
                    expr.pop(Kk, None)
        rs_prefix[J] = sol[J]
    played = [x for x in rows]
    n = len(played)
    n_valid_hit = sum(1 for x in played if x["in_valid"])
    ranks = [x["rank"] for x in played if x["rank"] is not None]
    n_top = sum(1 for x in ranks if x <= K_BEAM)
    print(f"\n===== {list(T)}: {n} playable steps of {len(acts)} actions")
    print(f"  truth action IN enumeration: {n_valid_hit}/{n}")
    print(f"  truth pivot in tied-target set: "
          f"{sum(1 for x in played if x['in_tied'])}/{n}")
    if ranks:
        print(f"  model rank of truth action: median={int(np.median(ranks))} "
              f"mean={np.mean(ranks):.0f} max={max(ranks)}  "
              f"in-top-{K_BEAM}: {n_top}/{len(ranks)}")
    out[T] = rows

with open(os.path.join(ROOT, "results/truth/rank_diag.pkl"), "wb") as f:
    pickle.dump(out, f)
print("\nsaved -> results/truth/rank_diag.pkl")
