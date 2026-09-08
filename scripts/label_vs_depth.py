#!/usr/bin/env python
"""Does the correct-set shrink with rollout depth?

In the emitter, lab = legal actions that are closure rows AND NOT ALREADY USED
on this state's path. So as a rollout consumes closure rows, lab mechanically
shrinks -- independent of the model. Validation has no such path dependence, so
if this effect is strong, comparing rollout top-1 against val_anyhit is unfair.
"""
import glob
import sys
from collections import defaultdict
import torch


def main():
    for label, pat in (('DAGGER', 'data_dagger/shard_*/train.pt'),
                       ('BASE  ', 'data/shard_*/train.pt')):
        legal = defaultdict(int); corr = defaultdict(int); n = defaultdict(int)
        for p in sorted(glob.glob(pat))[:40]:
            d = torch.load(p, map_location='cpu', weights_only=False)
            st = d['steps'].tolist()
            nv = d['num_valid_actions'].tolist()
            off = d['valid_label_offsets']
            cnt = (off[1:] - off[:-1]).tolist()
            for s, v, c in zip(st, nv, cnt):
                b = min(int(s) // 5 * 5, 40)
                n[b] += 1; legal[b] += v; corr[b] += c
        print(f'=== {label} : correct-set size vs step depth ===')
        print(f'  {"step":>8s} {"n":>9s} {"legal":>8s} {"correct":>8s} {"1 in":>7s}')
        for b in sorted(n):
            if n[b] < 200:
                continue
            L, C = legal[b] / n[b], corr[b] / n[b]
            print(f'  {b:>5d}-{b+4:<3d}{n[b]:>8,d} {L:>8.0f} {C:>8.2f} '
                  f'{L/C:>7.0f}')
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
