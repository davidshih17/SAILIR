"""Save the p=101 targets the CLASSICAL solver could not do, as a held-out set.

The 350 groups that died all died the same way: the memory watchdog fired while
building the shared linear system, before a single target was solved. 549 group
attempts breached 9 GB, 265 breached 19 GB. So the 5,398 targets they carried
are exactly the ones ordinary elimination cannot reach inside the memory budget.

WHY THEY MAKE A GOOD TEST SET, and what kind of test they are:

  * They are GUARANTEED disjoint from training. Training walks are generated
    from closures; these have no closure, so no walk, so no sample. This is
    held-out by construction, not by a random split.
  * They are hard in the RIGHT direction -- hard for the classical method,
    which is the thing the model is supposed to beat.
  * But there is NO ground truth to score against. No closure means no
    reference action sequence and no reference end state.

So they cannot be scored by imitation (top-1 against a recorded label). They CAN
be scored on the task itself: run the beam and ask whether the active bucket
drains -- i.e. whether the start integral was reduced by one weight level. That
check is self-verifying and needs no truth, which is precisely why this set is
usable at all.

Also emits the overlap with every other closure corpus on disk, because a target
solved in an EARLIER campaign is not a clean held-out candidate -- it may have
reached some older training corpus.
"""
import glob
import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
OUT = f'{D}/v4_p101_18k'


def tag(t):
    return '_'.join(str(x) for x in t)


def main():
    want, gof = {}, {}
    for f in sorted(glob.glob(f'{D}/groups_18k/group_*.json')):
        gid = int(os.path.basename(f)[6:-5])
        try:
            g = json.load(open(f))
        except Exception:
            continue
        for t in g.get('targets', g if isinstance(g, list) else []):
            want[tag(t)] = list(t)
            gof[tag(t)] = gid

    solved = set()
    for f in glob.glob(f'{OUT}/status_group_*.jsonl'):
        for line in open(f, errors='ignore'):
            line = line.strip()
            if not line:
                continue
            try:
                solved.add(json.loads(line)['tag'])
            except Exception:
                pass

    hard = sorted(set(want) - solved)
    print(f'requested {len(want):,}   solved {len(solved):,}   '
          f'HARD (unsolved) {len(hard):,}')

    # overlap with other corpora: a target solved elsewhere is not clean
    other = {}
    for corp in ('v1_18k', 'v3_p101_180'):
        s = set()
        for f in glob.glob(f'{D}/{corp}/**/*.json', recursive=True):
            if os.path.basename(f).startswith('group_'):
                continue
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if 'integral' in d and 'closure' in d:
                s.add(tag(d['integral']))
        other[corp] = s
        print(f'  overlap with {corp:14s} ({len(s):,} closures): '
              f'{len(s & set(hard)):,}')

    dirty = set()
    for s in other.values():
        dirty |= s
    clean = [h for h in hard if h not in dirty]
    print(f'\nCLEAN hard targets (no closure in ANY corpus): {len(clean):,}')

    p1 = f'{D}/hard_targets_p101.txt'
    with open(p1, 'w') as fh:
        for h in clean:
            fh.write(','.join(str(x) for x in want[h]) + '\n')
    p2 = f'{D}/hard_targets_p101.json'
    with open(p2, 'w') as fh:
        json.dump({'note': 'p=101 closure campaign: targets whose group died on '
                           'the memory watchdog before solving anything. No '
                           'ground-truth closure exists; score by whether the '
                           'beam drains the active bucket.',
                   'n_requested': len(want), 'n_solved': len(solved),
                   'n_hard': len(hard), 'n_clean': len(clean),
                   'targets': [want[h] for h in clean],
                   'group': [gof[h] for h in clean]}, fh)
    print(f'wrote {p1}')
    print(f'wrote {p2}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
