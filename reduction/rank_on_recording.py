#!/usr/bin/env python
"""Rank the LEX action at every step of a recorded walk. No replay.

WHY THIS REPLACES truth_rank_diagnostic4.py. That script rebuilt the closure
and walked it with a rule of its own -- "the target's own generator if usable,
else the first closure equation in postorder that eliminates it". That is a
THIRD policy, neither the minsumw walk ft_lex's states came from nor the lex
walk ft_lex enacts, so its states were off-distribution from step one and an
unknown share of the rank collapse it measured was the replay wandering rather
than anything a beam would see.

A recording already contains the walk. Every step carries the expression, the
substitution store, the full enumerated action list, and the index of the
chosen action within it. So the honest measurement is to stand in each recorded
state, ask the model to rank the actions, and report where the recorded choice
landed. No policy of ours enters at all.

That also makes the comparison exact rather than approximate: point this at the
same recording with two checkpoints and both see byte-identical states, so any
difference is the model.

  ft_lex        on lex_corpus  -> what ft_lex does on states it never trained on
  ft_onpolicy   on lex_corpus  -> the same states, trained on their own walk

Sampling: --every N scores one step in N. A full 1,200-step recording is 1,200
forward passes over action lists that grow into the thousands, so scoring every
step of every target is not free; the rank profile is smooth enough that every
10th step resolves the depth trend.

Usage:
  rank_on_recording.py --recording <rec.jsonl> --checkpoint <model.pt> [--every 10]
"""
import os, sys, json, argparse, pickle

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))

ap = argparse.ArgumentParser()
ap.add_argument('--recording', required=True)
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--outdir', required=True)
ap.add_argument('--tag', default='')
ap.add_argument('--every', type=int, default=10)
ap.add_argument('--k-beam', type=int, default=20)
ap.add_argument('--cpus', type=int, default=4)
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

sys.argv = [sys.argv[0], '--v7-cpus', str(args.cpus)]
import beam_search_v7 as bs7                       # batch prep only

# Architecture and prime come from the checkpoint, never defaulted -- a wrong
# prime was a two-month silent bug once.
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
print(f"[model] {args.checkpoint}", flush=True)
print(f"[model] epoch={ck.get('epoch')} prime={cka['prime']} "
      f"select_on={cka.get('select_on')} val={ck.get('val_metrics')}", flush=True)
print(f"[data]  {args.recording}", flush=True)


def _expr(j):
    return {tuple(k): v for k, v in j}


def _subs(j):
    return {tuple(k): {tuple(k2): v2 for k2, v2 in tail} for k, tail in j}


rows = []
n_seen = 0
with open(args.recording) as fh:
    for line in fh:
        d = json.loads(line)
        n_seen += 1
        if (d['step'] % args.every) != 0:
            continue
        valid = [(int(op), tuple(dl)) for op, dl in d['valid_actions']]
        if not valid:
            continue
        lex_idx = int(d['chosen_action_idx'])
        expr = _expr(d['expr'])
        rs = _subs(d['subs'])
        J = tuple(d['target'])
        bd = [(expr, rs, valid, tuple(d['sector_mask']), J)]
        b = bs7.prepare_batched_input_v5_dummy(bd, 'cpu',
                                               max_actions=len(valid))
        with torch.no_grad():
            _, probs = model(
                b['expr_integrals'], b['expr_coeffs'], b['expr_mask'],
                b['sub_keys'], b['sub_repl_ints'], b['sub_repl_coeffs'],
                b['sub_repl_mask'], b['sub_mask'],
                b['action_ibp_ops'], b['action_deltas'], b['action_mask'],
                b['sector_mask'], b['target_integral'],
            )
        p = probs[0, :len(valid)].numpy()
        order = np.argsort(-p)
        pos = {int(v): i for i, v in enumerate(order)}
        lexrank = pos[lex_idx] + 1
        # valid_label_idxs is recorded, so the "any correct action" rank costs
        # nothing extra here -- unlike the replay, which had to rediscover it.
        cor = [int(i) for i in d.get('valid_label_idxs') or []]
        anyrank = (min(pos[i] for i in cor if i in pos) + 1) if cor else None
        rows.append(dict(step=int(d['step']), lexrank=lexrank,
                         anyrank=anyrank, n_valid=len(valid),
                         n_correct=len(cor), p_top=float(p[order[0]]),
                         p_lex=float(p[lex_idx])))
        print(f"  step {d['step']:5d} lexrank {lexrank:6d} "
              f"anyrank {anyrank} n_valid {len(valid):5d} "
              f"n_correct {len(cor):4d} p_top {p[order[0]]:.5f}", flush=True)

L = np.array([r['lexrank'] for r in rows], float)
K = args.k_beam
print(f"\n===== {os.path.basename(args.recording)}")
print(f"  walk length {n_seen} steps, scored {len(rows)} (every {args.every})")
if len(L):
    print(f"  LEX rank: median {int(np.median(L))}  p90 {np.percentile(L, 90):.0f}"
          f"  max {int(L.max())}")
    print(f"  LEX-in-top-{K}: {np.mean(L <= K):.4f}   LEX-top1: {np.mean(L == 1):.4f}")
    e = np.array([r['step'] for r in rows]) <= 40
    l = np.array([r['step'] for r in rows]) >= 100
    if e.any() and l.any():
        print(f"  early(step<=40) {np.mean(L[e] <= K):.3f}   "
              f"late(step>=100) {np.mean(L[l] <= K):.3f}")
    A = [r['anyrank'] for r in rows if r['anyrank'] is not None]
    if A:
        A = np.array(A, float)
        print(f"  ANY-correct-in-top-{K}: {np.mean(A <= K):.4f}  "
              f"median {int(np.median(A))}")

os.makedirs(args.outdir, exist_ok=True)
tag = args.tag or os.path.basename(args.recording).replace('.jsonl', '')
with open(os.path.join(args.outdir, f'rank_{tag}.pkl'), 'wb') as f:
    pickle.dump(dict(rows=rows, recording=args.recording,
                     checkpoint=args.checkpoint, walk_len=n_seen,
                     every=args.every, k_beam=K), f)
print(f"saved -> {args.outdir}/rank_{tag}.pkl", flush=True)
