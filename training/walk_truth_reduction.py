"""Walk a TRUE reduction from the validation set step by step and report how
each model ranks the correct actions.

Aggregate metrics say what happens on average; this says what happens along a
single real reduction, which is what the beam actually has to survive. At every
step we ask: where does the best correct action rank among all legal ones, how
many WRONG actions outrank it, and how confident is the model.

Trajectories are grouped by `start_target_integrals` (the integral being
reduced); `steps` orders them and `steps_remaining` counts down to 1.

Usage:
  python training/walk_truth_reduction.py \
      --topology topology_input/gravity3L --shards_dir data \
      --checkpoint bce=checkpoints/gravity3L_p101_bce/best_model.pt \
      --checkpoint ce=checkpoints/gravity3L_p101_scratch/best_model.pt
"""
import argparse, collections, sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / 'sailir'))

from train_classifier import PackedDatasetV5, make_collate_fn, model_forward
from sailir.topology import Topology
from classifier_nosubs import IBPActionClassifierNoSubs


def load_model(path, topo, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck['args']
    act = a.get('score_activation', 'softmax')
    m = IBPActionClassifierNoSubs(
        embed_dim=a['embed_dim'], n_heads=a['n_heads'],
        n_expr_layers=a['n_expr_layers'], n_cross_layers=a['n_cross_layers'],
        n_subs_layers=a.get('n_subs_layers', 2), prime=a['prime'],
        n_indices=topo.n_indices, n_denominators=topo.n_denominators,
        n_ibp_ops=topo.n_actions, score_activation=act,
    ).to(device)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m, ck['epoch'], act, a['prime']


def find_longest(shards_dir, n_scan):
    """Return (shard_path, sample_indices_in_step_order, n_steps) of the longest val trajectory."""
    dirs = sorted(Path(shards_dir).glob('shard_*'), key=lambda p: int(p.name.split('_')[-1]))
    best = (None, None, -1)
    for d in dirs[:n_scan]:
        f = d / 'val.pt'
        if not f.is_file():
            continue
        s = torch.load(f, weights_only=False)
        keys = [tuple(r.tolist()) for r in s['start_target_integrals']]
        groups = collections.defaultdict(list)
        for i, k in enumerate(keys):
            groups[k].append(i)
        for k, idx in groups.items():
            if len(idx) > best[2]:
                idx = sorted(idx, key=lambda i: int(s['steps'][i]))
                best = (f, idx, len(idx))
        del s
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topology', required=True)
    ap.add_argument('--shards_dir', required=True)
    ap.add_argument('--checkpoint', action='append', required=True,
                    help='name=path, repeatable')
    ap.add_argument('--n_scan_shards', type=int, default=40)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    topo = Topology.from_dir(args.topology)
    dev = args.device

    shard, idx, n = find_longest(args.shards_dir, args.n_scan_shards)
    print(f"Longest val trajectory in first {args.n_scan_shards} shards: "
          f"{n} steps, from {shard}\n")
    s = torch.load(shard, weights_only=False)
    ds = PackedDatasetV5(s)
    collate = make_collate_fn(topo.n_indices, topo.n_denominators)

    models = {}
    for spec in args.checkpoint:
        name, path = spec.split('=', 1)
        m, ep, act, pr = load_model(path, topo, dev)
        models[name] = m
        print(f"  {name:5s}: {path}  (epoch {ep}, {act}, prime={pr})")
    print()

    for name, m in models.items():
        print("=" * 92)
        print(f"MODEL {name}   [score = {m.score_activation}]")
        print("=" * 92)
        print(f"{'step':>4} {'legal':>6} {'ncor':>5} | {'rank of best':>12} "
              f"{'#wrong above':>12} | {'best_cor':>9} {'best_wrong':>10} "
              f"{'margin':>8} | {'all correct ranks':<24}")
        print("-" * 92)
        n_top1 = 0
        ranks_all = []
        for pos, i in enumerate(idx):
            b = collate([ds[i]])
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.no_grad():
                logits, scores = model_forward(m, b)
            act_m = b['action_mask'][0]
            cor = b['label_mask'][0] & act_m
            inc = (~b['label_mask'][0]) & act_m
            sc = scores[0]
            n_legal = int(act_m.sum())
            n_cor = int(cor.sum())
            if n_cor == 0:
                continue
            best_cor = sc[cor].max().item()
            best_wrong = sc[inc].max().item() if int(inc.sum()) else float('nan')
            # rank among LEGAL actions (1 = highest scoring)
            legal_scores = sc[act_m]
            rank_best = int((legal_scores > best_cor).sum()) + 1
            n_wrong_above = int((sc[inc] > best_cor).sum()) if int(inc.sum()) else 0
            cor_ranks = sorted(int((legal_scores > v).sum()) + 1 for v in sc[cor].tolist())
            ranks_all.append(rank_best)
            n_top1 += (rank_best == 1)
            rs = ",".join(str(r) for r in cor_ranks[:8]) + ("..." if len(cor_ranks) > 8 else "")
            print(f"{pos:>4} {n_legal:>6} {n_cor:>5} | {rank_best:>12} {n_wrong_above:>12} | "
                  f"{best_cor:>9.4f} {best_wrong:>10.4f} {best_cor-best_wrong:>+8.4f} | {rs:<24}")
        print("-" * 92)
        print(f"  steps where a correct action ranked #1 : {n_top1}/{len(ranks_all)} "
              f"({n_top1/max(len(ranks_all),1):.1%})")
        print(f"  best-correct rank: median {sorted(ranks_all)[len(ranks_all)//2]}, "
              f"max {max(ranks_all)}")
        print(f"  steps where best correct is inside top-20 : "
              f"{sum(r<=20 for r in ranks_all)}/{len(ranks_all)}")
        print()


if __name__ == '__main__':
    main()
