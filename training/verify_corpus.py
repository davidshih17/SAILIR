"""
Pre-flight verification for a packed corpus before a long training run.

Checks, across a sample of shards:
  1. Coefficient range is consistent with the claimed prime (all coeffs in
     [0, p) after the signed->unsigned convention). A p=1009 corpus fed to a
     --prime 101 run is silently wrong, so this is worth 30 seconds.
  2. Tensor widths match the topology (n_indices, n_denominators).
  3. Labels index inside the per-sample action set.
  4. Reports train/val sample counts so the epoch size is known up front.

Usage:
    python training/verify_corpus.py --shards_dir data \
        --topology topology_input/gravity3L --prime 101 --n_shards 20
"""

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards_dir', required=True)
    ap.add_argument('--topology', required=True)
    ap.add_argument('--prime', type=int, required=True)
    ap.add_argument('--n_shards', type=int, default=20,
                    help='How many shards to sample (evenly spaced).')
    args = ap.parse_args()

    from sailir.topology import Topology
    topo = Topology.from_dir(args.topology)
    print(f"Topology {args.topology}: n_indices={topo.n_indices} "
          f"n_denominators={topo.n_denominators} n_actions={topo.n_actions}", flush=True)

    root = Path(args.shards_dir)
    shard_dirs = sorted(root.glob('shard_*'), key=lambda p: int(p.name.split('_')[-1]))
    print(f"Found {len(shard_dirs)} shard dirs", flush=True)
    if not shard_dirs:
        sys.exit("No shard_* dirs found")

    # Evenly spaced sample so we cover the whole corpus, not just the head.
    step = max(1, len(shard_dirs) // args.n_shards)
    picked = shard_dirs[::step][: args.n_shards]

    problems = []
    tot = {'train': 0, 'val': 0}
    coeff_min, coeff_max = None, None
    n_ge_prime = 0
    n_coeffs_seen = 0

    for d in picked:
        for split in ('train', 'val'):
            f = d / f'{split}.pt'
            if not f.is_file():
                problems.append(f"{f} missing")
                continue
            s = torch.load(f, weights_only=False)
            n = len(s['labels'])
            tot[split] += n

            # --- width checks ---
            if s['target_integrals'].shape[1] != topo.n_indices:
                problems.append(f"{f}: target_integrals width "
                                f"{s['target_integrals'].shape[1]} != n_indices {topo.n_indices}")
            if s['sector_masks'].shape[1] != topo.n_denominators:
                problems.append(f"{f}: sector_masks width "
                                f"{s['sector_masks'].shape[1]} != n_denominators {topo.n_denominators}")
            if s['action_deltas'].shape[1] != topo.n_indices:
                problems.append(f"{f}: action_deltas width "
                                f"{s['action_deltas'].shape[1]} != n_indices {topo.n_indices}")

            # --- coefficient range (prime consistency) ---
            c = s['expr_coeffs'].to(torch.int64)
            cmin, cmax = int(c.min()), int(c.max())
            coeff_min = cmin if coeff_min is None else min(coeff_min, cmin)
            coeff_max = cmax if coeff_max is None else max(coeff_max, cmax)
            n_ge_prime += int((c >= args.prime).sum())
            n_coeffs_seen += c.numel()

            # --- label validity ---
            # labels index into that sample's action list (0 <= label < n_actions_i)
            n_valid = s['num_valid_actions'].to(torch.int64)
            lab = s['labels'].to(torch.int64)
            bad = int(((lab < 0) | (lab >= n_valid)).sum())
            if bad:
                problems.append(f"{f}: {bad} labels outside their action set")

            del s

    print(f"\nSampled {len(picked)} shards:", flush=True)
    print(f"  train samples: {tot['train']:,}", flush=True)
    print(f"  val   samples: {tot['val']:,}", flush=True)
    if tot['train']:
        scale = len(shard_dirs) / len(picked)
        print(f"  => projected full corpus: train ~{int(tot['train']*scale):,}  "
              f"val ~{int(tot['val']*scale):,}", flush=True)

    print(f"\nCoefficient range: [{coeff_min}, {coeff_max}] over {n_coeffs_seen:,} coeffs", flush=True)
    print(f"  coeffs >= prime({args.prime}): {n_ge_prime:,}", flush=True)
    if n_ge_prime:
        problems.append(f"{n_ge_prime} coefficients >= --prime {args.prime}: "
                        f"corpus prime does NOT look like {args.prime}")
    elif coeff_max is not None and coeff_max < args.prime // 2:
        print(f"  WARNING: max coeff {coeff_max} is well below {args.prime}; "
              f"consistent with {args.prime} but also with a smaller prime.", flush=True)

    print("", flush=True)
    if problems:
        print("PROBLEMS:", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)
        sys.exit(1)
    print("CORPUS VERIFICATION PASSED", flush=True)


if __name__ == '__main__':
    main()
