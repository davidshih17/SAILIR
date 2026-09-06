"""Measure the positive/negative balance of the multi-label action targets.

Needed to set pos_weight for a BCE objective: with ~N_pos correct actions out
of ~N_act legal ones per state, plain BCE is dominated by the negatives.
"""
import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards_dir', required=True)
    ap.add_argument('--n_shards', type=int, default=12)
    args = ap.parse_args()

    dirs = sorted(Path(args.shards_dir).glob('shard_*'),
                  key=lambda p: int(p.name.split('_')[-1]))
    step = max(1, len(dirs) // args.n_shards)
    picked = dirs[::step][: args.n_shards]

    n_samples = 0
    tot_actions = 0
    tot_valid = 0
    per_sample_actions = []
    per_sample_valid = []

    for d in picked:
        s = torch.load(d / 'train.pt', weights_only=False)
        nval = s['num_valid_actions'].to(torch.int64)          # legal actions per state
        n = len(s['labels'])
        n_samples += n
        tot_actions += int(nval.sum())
        per_sample_actions.append(nval)

        # count of CORRECT actions per state, from the multi-label field
        if 'valid_label_offsets' in s:
            off = s['valid_label_offsets'].to(torch.int64)
            cnt = off[1:] - off[:-1]
        else:                                                   # fallback: single label
            cnt = torch.ones(n, dtype=torch.int64)
        tot_valid += int(cnt.sum())
        per_sample_valid.append(cnt)
        del s

    A = torch.cat(per_sample_actions).float()
    V = torch.cat(per_sample_valid).float()

    print(f"shards sampled       : {len(picked)}")
    print(f"samples              : {n_samples:,}")
    print()
    print(f"legal actions/state  : mean {A.mean():.1f}  median {A.median():.0f}  "
          f"min {A.min():.0f}  max {A.max():.0f}")
    print(f"CORRECT actions/state: mean {V.mean():.2f}  median {V.median():.0f}  "
          f"min {V.min():.0f}  max {V.max():.0f}")
    print()
    pos_rate = tot_valid / max(tot_actions, 1)
    print(f"total legal actions  : {tot_actions:,}")
    print(f"total correct        : {tot_valid:,}")
    print(f"positive rate        : {pos_rate:.4%}")
    print(f"neg:pos ratio        : {(1-pos_rate)/pos_rate:.1f} : 1")
    print()
    print(f"=> pos_weight (neg/pos) = {(tot_actions-tot_valid)/max(tot_valid,1):.1f}")
    # how often is the single recorded label the ONLY correct one?
    print(f"states with exactly 1 correct action: {(V==1).float().mean():.2%}")
    print(f"states with >1 correct action      : {(V>1).float().mean():.2%}")


if __name__ == "__main__":
    main()
