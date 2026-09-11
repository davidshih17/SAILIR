#!/usr/bin/env python
"""MODEL-RANK DIAGNOSTIC v4 -- one target per process, ft_lex, closure from the
truth engine.

THE QUESTION. Three integrals -- (2,1,1,1,1,1,1), (3,1,1,1,1,1,1),
(4,1,1,1,1,1,1) -- have truth reductions of only 876 / 678 / 1,216 steps, ALL
THREE sit in the TRAIN split of the corpus ft_lex was trained on, and the beam
is 2-5x past those path lengths without closing. So the model was shown the
correct action at every step of exactly these reductions and the search still
fails. Two explanations, and they demand opposite fixes:

  (1) MODEL. From the states the beam actually visits, the model cannot rank
      the truth action highly -- training on the path did not transfer off it.
  (2) SEARCH. The model ranks it fine, and the beam sort throws the lineage
      away.

This script separates them. It walks the truth trajectory in WORKER form and,
before playing each step, asks the model to rank every enumerated valid action.
If the truth action sits inside top-K=20 (what the beam expands) throughout,
the model is doing its job and the defect is in the beam -- reading (2). If the
rank collapses partway, reading (1).

WHY WORKER FORM AND NOT THE RECORDING. `worker_replay` emits the dependency
closure dependencies-first, leaving the expression as {T} until the very last
action -- that is not what the beam experiences. The loop below is genuine
worker dynamics instead: the expression evolves, each step targets the highest
total-lex-weight active non-master, and the action played is a closure equation
whose RAW form, resolved by the CURRENT substitution store, still contains that
target (preferring the target's own generator). Same rule the worker uses. The
1.3-4.0 GB jsonl recordings are therefore not read at all -- the closure is
rebuilt, which is both cheaper and guaranteed consistent with this engine.

CAVEAT, STATED UP FRONT. Ranking here is over `enumerate_valid_actions`, while
v8 ranks over `enumerate_valid_actions_with_indirect_cache_packed`, which can
surface additional indirect actions. Fewer distractors means ranks measured
here are optimistic. That asymmetry only matters in one direction: a BAD rank
here is conclusive (it would be no better with more competitors), a good rank
needs the full enumerator before it can be trusted. `n_valid` is logged per
step so the two enumerations can be compared against the beam's own logs.

Usage:  truth_rank_diagnostic4.py --integral 3,1,1,1,1,1,1,0,0,0,0,0,0,0,0
"""
import os, sys, pickle, argparse, time

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))

ap = argparse.ArgumentParser()
ap.add_argument('--integral', required=True)
ap.add_argument('--checkpoint',
                default=os.path.join(ROOT, "checkpoints/gravity3L_dots_lex/best_model.pt"))
ap.add_argument('--outdir', required=True)
ap.add_argument('--k-beam', type=int, default=20)
ap.add_argument('--cpus', type=int, default=4)
ap.add_argument('--anyhit-every', type=int, default=10,
                help='every Nth step, also find EVERY closure equation that '
                     'eliminates the current target and report the best rank '
                     'among them (anyhit). Sampled rather than done every step '
                     'because it costs one resolve per closure equation -- '
                     'O(800-1700) resolves -- versus the early-exit search the '
                     'rest of the loop uses. 0 disables.')
args = ap.parse_args()

import numpy as np
import torch
torch.set_num_threads(args.cpus)

from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc

P = 1009
ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))
ibp_env.set_prime(P)
ibp_env.set_paper_masters_only(True)
from canonical_masters import apply_canonical_masters
apply_canonical_masters()
from sailir.ibp_env import enumerate_valid_actions, is_master, get_raw_equation
from beam_search_utils import get_sector_mask
from truth_engine import TruthEngine, sector_of

sys.argv = [sys.argv[0], '--v7-cpus', str(args.cpus)]   # v7 import-time CPU pin
import beam_search_v7 as bs7                            # batch prep only

# ---- model: architecture read from the checkpoint, never defaulted. A wrong
# prime here was a two-month silent bug once; nothing about the model is
# assumed.
ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
cka = ck.get('args') or {}
if hasattr(cka, '__dict__'):
    cka = vars(cka)
from sailir.classifier_nosubs import IBPActionClassifierNoSubs
topo = Topology.from_dir(_tc.TOPO_DIR)
model = IBPActionClassifierNoSubs(
    embed_dim=cka['embed_dim'], n_heads=cka['n_heads'],
    n_expr_layers=cka['n_expr_layers'], n_cross_layers=cka['n_cross_layers'],
    prime=cka['prime'], n_indices=topo.n_indices,
    n_denominators=topo.n_denominators, n_ibp_ops=topo.n_actions)
model.load_state_dict(ck['model_state_dict'])
model.eval()
print(f"[model] {args.checkpoint}\n[model] epoch={ck.get('epoch')} "
      f"prime={cka['prime']} embed={cka['embed_dim']} "
      f"select_on={cka.get('select_on')} val={ck.get('val_metrics')}", flush=True)

T = tuple(int(x) for x in args.integral.split(','))
triv = os.path.join(_tc.TOPO_DIR, "kira_validate/sectormappings/GR/trivialsector")
eng = TruthEngine(_tc.TOPO_DIR, triv)

t0 = time.time()
print(f"[closure] building the dependency closure of {list(T)} ...", flush=True)
wr = eng.worker_replay(T, verbose=True)
gens = {J: (op, tuple(x + y for x, y in zip(J, delta)))
        for J, op, delta in wr['actions']}
all_eqs = [(J, op, seed) for J, (op, seed) in gens.items()]
print(f"[closure] {len(gens)} actions in {time.time() - t0:.1f}s", flush=True)


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


kT = eng.tkey(T)
tsec_mask = get_sector_mask(T)
expr = {T: 1 % P}
rs = {}
rows = []
status = 'RAN_OUT'
for step in range(3 * len(gens) + 10):
    for J in [j for j in list(expr)
              if sector_of(j) != 0 and sector_of(j) in eng.trivial]:
        expr.pop(J)
    nm = [J for J in expr if expr[J] % P and not is_master(J)
          and sector_of(J) != 0]
    active = [J for J in nm if eng.tkey(J) <= kT]
    if not active:
        status = 'DRAINED'
        print(f"  DRAINED at step {step}", flush=True)
        break
    J = min(nm, key=eng.tkey)
    # the truth action: J's own generator if worker-valid now, else any closure
    # equation whose rs-resolved raw still contains J
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
        status = 'CLOSURE_INSUFFICIENT'
        print(f"  STOP at step {step}: no closure equation eliminates "
              f"{list(J)} under worker-dynamics order", flush=True)
        break
    op, seed, res = pick
    delta = tuple(x - y for x, y in zip(seed, J))
    # ---- rank the truth action among everything the model could pick ----
    valid = enumerate_valid_actions(J, rs, eng.env.ibp_t, eng.env.li_t,
                                    eng.env.shifts, 'subsector')
    try:
        vidx = valid.index((op, delta))
    except ValueError:
        vidx = -1
    # ANYHIT. The rank above is of ONE designated action, but ~27-30 actions
    # per state are equally correct, so a bad rank there can mean nothing worse
    # than "the model preferred a different valid reduction". What the beam
    # actually needs is for SOME correct action to be inside the top K it
    # expands. Collect every closure equation that eliminates J -- that is the
    # correct-action set restricted to this closure -- and take the best rank.
    # Also identify the LEX action -- min (op, delta) among the correct ones.
    # That is the label ft_lex was actually trained to emit, so its rank is the
    # only one directly comparable to the model's val_top20, and the one to
    # judge the model by. The first-found action above is arbitrary.
    all_idx, lex_idx = [], None
    if args.anyhit_every and step % args.anyhit_every == 0:
        cor = []
        for p2, op2, seed2 in cand_list:
            raw2 = get_raw_equation(eng.env.ibp_t, eng.env.li_t, op2, seed2)
            res2 = resolve(raw2, rs)
            if J in res2 and res2[J] % P:
                d2 = tuple(x - y for x, y in zip(seed2, J))
                try:
                    i2 = valid.index((op2, d2))
                except ValueError:
                    continue
                cor.append(((op2, d2), i2))
        if cor:
            seen = {}
            for key, i2 in cor:
                seen.setdefault(key, i2)
            all_idx = sorted(set(seen.values()))
            lex_idx = seen[min(seen)]
    rank = p_t = p_top = None
    anyrank = n_correct = lexrank = None
    top_ok = None
    top1_valid = None
    if vidx >= 0 and valid:
        bd = [(expr, rs, valid, tsec_mask, J)]
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
        rowp = probs[0, :len(valid)].numpy()
        order = np.argsort(-rowp)
        rank = int(np.where(order == vidx)[0][0]) + 1
        p_t, p_top = float(rowp[vidx]), float(rowp[order[0]])
        # ---- THE population claim, checked on EVERY step ----
        # "Is the model's top pick a valid reducing action?" needs only ONE
        # resolve: take its (op, delta), build the raw equation, substitute the
        # current store, and see whether the target survives with a non-zero
        # coefficient. That is the definition of a usable action, and it does
        # not depend on closure membership -- so it is both cheaper than the
        # anyhit scan AND a strictly broader test (an action outside this
        # target's closure can still be perfectly valid).
        op1, d1 = valid[int(order[0])]
        raw1 = get_raw_equation(eng.env.ibp_t, eng.env.li_t, op1,
                                tuple(x + y for x, y in zip(J, d1)))
        res1 = resolve(raw1, rs)
        top1_valid = bool(J in res1 and res1[J] % P)
        if all_idx:
            pos = {int(v): i for i, v in enumerate(order)}
            anyrank = min(pos[i] for i in all_idx if i in pos) + 1
            n_correct = len(all_idx)
            if lex_idx is not None and lex_idx in pos:
                lexrank = pos[lex_idx] + 1
            # is the model's OWN confident pick actually a correct action?
            top_ok = int(order[0]) in set(all_idx)
    rows.append(dict(step=step, J=J, op=op, delta=delta, n_valid=len(valid),
                     in_valid=vidx >= 0, rank=rank, p_truth=p_t, p_top=p_top,
                     anyrank=anyrank, n_correct=n_correct, lexrank=lexrank,
                     top_ok=(top_ok if all_idx else None),
                     top1_valid=top1_valid,
                     expr_size=len(expr), n_active=len(active),
                     n_subs=len(rs)))
    print(f"  step {step:4d} J={list(J)} valid={len(valid):4d} "
          f"in_valid={vidx >= 0} rank={rank} "
          f"p_truth={None if p_t is None else round(p_t, 5)} "
          f"p_top={None if p_top is None else round(p_top, 5)} "
          f"anyrank={anyrank} lexrank={lexrank} n_correct={n_correct} "
          f"top_ok={top_ok} top1_valid={top1_valid} "
          f"expr={len(expr)} active={len(active)} subs={len(rs)}", flush=True)
    # ---- play (worker semantics) ----
    co = expr.pop(J)
    inv = pow(res[J], P - 2, P)
    tail = {K: (-c * inv) % P for K, c in res.items() if K != J and c % P}
    rs[J] = tail
    for Kk, cK in tail.items():
        v = (expr.get(Kk, 0) + co * cK) % P
        if v:
            expr[Kk] = v
        else:
            expr.pop(Kk, None)
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
K = args.k_beam
print(f"\n===== SUMMARY {list(T)}  status={status}")
print(f"  closure actions      : {len(gens)}")
print(f"  worker steps walked  : {n}")
print(f"  truth action enumerated: {sum(1 for x in rows if x['in_valid'])}/{n}")
if ranks:
    intop = sum(1 for r in ranks if r <= K)
    print(f"  rank median={int(np.median(ranks))} mean={np.mean(ranks):.1f} "
          f"max={max(ranks)}")
    print(f"  IN TOP-{K}: {intop}/{len(ranks)} = {intop / len(ranks):.4f}")
    # where it first falls out of the beam's reach -- the step the search
    # could first have lost the truth lineage
    first_out = next((x['step'] for x in rows
                      if x['rank'] is not None and x['rank'] > K), None)
    print(f"  first step with rank > {K}: {first_out}")
    for q in (50, 75, 90, 95, 99):
        print(f"    p{q} rank = {np.percentile(ranks, q):.0f}")
# ANYHIT is the number that decides model-vs-search: it asks whether the beam
# had ANY correct continuation available inside the K it expands, which is the
# condition deployment actually needs.
anyr = [x["anyrank"] for x in rows if x.get("anyrank") is not None]
ncor = [x["n_correct"] for x in rows if x.get("n_correct") is not None]
if anyr:
    a = np.array(anyr)
    print(f"\n  --- ANYHIT (sampled every {args.anyhit_every} steps, "
          f"n={len(a)}) ---")
    print(f"  correct actions per state: median {int(np.median(ncor))} "
          f"min {min(ncor)} max {max(ncor)}")
    print(f"  best-correct rank: median={int(np.median(a))} "
          f"p90={np.percentile(a, 90):.0f} max={a.max()}")
    print(f"  ANYHIT-{K}: {np.mean(a <= K):.4f}   ANYHIT-1: {np.mean(a == 1):.4f}")
    lx = [x["lexrank"] for x in rows if x.get("lexrank") is not None]
    if lx:
        l = np.array(lx)
        print(f"  LEX action (what ft_lex was TRAINED to emit), n={len(l)}: "
              f"median={int(np.median(l))} p90={np.percentile(l, 90):.0f} "
              f"max={l.max()}")
        print(f"  LEX-in-top-{K}: {np.mean(l <= K):.4f}   LEX-top1: "
              f"{np.mean(l == 1):.4f}   (ft_lex val_top20 = 0.9891, "
              f"val_top1 = 0.8001)")
    t1 = [x["top1_valid"] for x in rows if x.get("top1_valid") is not None]
    if t1:
        bad = [x["step"] for x in rows if x.get("top1_valid") is False]
        print(f"\n  *** TOP-1 VALIDITY, EVERY STEP (n={len(t1)}, not sampled) ***")
        print(f"  model's #1 action is a usable reducing action: "
              f"{np.mean(t1):.4f}  ({sum(t1)}/{len(t1)})")
        print(f"  steps where it was NOT: {bad[:25]}"
              f"{' ...' if len(bad) > 25 else ''}")
    tk = [x["top_ok"] for x in rows if x.get("top_ok") is not None]
    if tk:
        print(f"  model's own #1 pick is a CORRECT action: "
              f"{np.mean(tk):.4f}  (n={len(tk)})")

os.makedirs(args.outdir, exist_ok=True)
tag = '_'.join(str(x) for x in T)
with open(os.path.join(args.outdir, f'rank4_{tag}.pkl'), 'wb') as f:
    pickle.dump(dict(target=T, rows=rows, status=status,
                     n_closure=len(gens), checkpoint=args.checkpoint,
                     k_beam=K), f)
print(f"\nsaved -> {args.outdir}/rank4_{tag}.pkl", flush=True)
