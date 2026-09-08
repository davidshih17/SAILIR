#!/usr/bin/env python
"""Compare label geometry: base corpus (on-path) vs DAgger shards (off-path).

A drop in "top-1 hits a correct action" can come from two different places:
  (a) the model being worse on a shifted distribution, or
  (b) the STATES being harder targets -- more legal actions, fewer correct ones,
      so the needle is smaller regardless of the model.
This separates them by measuring the geometry directly from the packed shards.
"""
import glob
import sys
import torch


def stats(paths, label, limit=40):
    n = 0
    legal = 0
    corr = 0
    for p in paths[:limit]:
        d = torch.load(p, map_location='cpu', weights_only=False)
        nv = d['num_valid_actions']
        off = d['valid_label_offsets']
        cnt = off[1:] - off[:-1]
        n += int(nv.numel())
        legal += int(nv.sum())
        corr += int(cnt.sum())
    print(f'{label}')
    print(f'  samples                 : {n:,}   ({min(limit,len(paths))} shards)')
    print(f'  mean legal actions      : {legal/n:.1f}')
    print(f'  mean CORRECT actions    : {corr/n:.2f}')
    print(f'  positive rate           : {corr/legal:.3%}')
    print(f'  1 / positive rate       : 1 correct in {legal/corr:.0f} legal')
    return legal / n, corr / n, corr / legal


def main():
    b = sorted(glob.glob('data/shard_*/train.pt'))
    a = sorted(glob.glob('data_dagger/shard_*/train.pt'))
    lb, cb, pb = stats(b, 'BASE CORPUS (truth-path states)')
    print()
    la, ca, pa = stats(a, 'DAGGER (states the model itself reaches)')
    print()
    print('RATIO (dagger / base)')
    print(f'  legal actions   : {la/lb:.2f}x')
    print(f'  correct actions : {ca/cb:.2f}x')
    print(f'  positive rate   : {pa/pb:.2f}x   <- how much smaller the needle is')
    return 0


if __name__ == '__main__':
    sys.exit(main())
