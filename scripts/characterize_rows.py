#!/usr/bin/env python
"""How hard are the emitted DAgger rows?

Every emitted row is a state where the model's top-1 was NOT in the expert's
correct set. This reports how thin that correct set is against how many legal
actions the model had to choose from -- i.e. how much of a needle each row is.
"""
import json
import sys
from collections import Counter


def main(path):
    n = 0
    lab = Counter()
    valid = []
    labs = []
    for ln in open(path):
        d = json.loads(ln)
        n += 1
        k = len(d['valid_label_idxs'])
        lab[k] += 1
        labs.append(k)
        valid.append(d['num_valid_actions'])
    valid.sort(); labs.sort()

    def pct(a, p):
        return a[min(len(a) - 1, int(len(a) * p))]

    print(f'rows: {n:,}')
    print(f'  legal actions/state   min/med/p90/max : '
          f'{valid[0]} / {pct(valid,.5)} / {pct(valid,.9)} / {valid[-1]}')
    print(f'  CORRECT actions/state min/med/p90/max : '
          f'{labs[0]} / {pct(labs,.5)} / {pct(labs,.9)} / {labs[-1]}')
    tot = sum(valid); totl = sum(labs)
    print(f'  mean legal {tot/n:.1f}   mean correct {totl/n:.2f}   '
          f'-> {totl/tot:.3%} of legal actions are correct')
    print()
    print('  distribution of correct-set size:')
    for k in sorted(lab)[:8]:
        print(f'    {k:3d} correct action(s): {lab[k]:7,d}  ({lab[k]/n:5.1%})')
    big = sum(v for k, v in lab.items() if k > 7)
    if big:
        print(f'    >7 correct actions : {big:7,d}  ({big/n:5.1%})')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
