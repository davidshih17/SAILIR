"""CANONICAL comparison: p=101 vs the K=1000 ftcull baseline.

USE THIS ONE. p101_vs_k1000_snapshot.py was DELETED -- it keyed membership
off directories and reported 28 solves for a 25-target set, inflating every
baseline test25 figure it produced.

p=101 vs the K=1000 ftcull baseline, with membership defined by the TARGET
LISTS rather than by which directory a log happens to live in.

The v1 script keyed baseline test25 off `beamexp25/logs` + `beamexp/logs`. But
`beamexp/` is a MIXTURE of campaigns (the EXPDIR incident), so its prob40 tree
holds targets that are not in test25 -- which is why it reported 28 solves for a
25-target set and inflated every baseline test25 figure.

Here a solve counts only if the target tag is in the campaign's own jobs file.
Anything else is foreign and dropped, whatever tree it sits in.
"""
import glob
import os
import re
import sys
import time

F = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/truth/finetune'
RE = re.compile(r'\[v7-worker\] SUCCESS in ([\d.]+)s')
OUT = re.compile(r'-> (\S+\.pkl)')


def members(jobs):
    """Target tags for a campaign, from its jobs file (tag, mode, bw)."""
    out = set()
    for line in open(os.path.join(F, jobs)):
        line = line.strip()
        if line:
            out.add(line.split(',')[0].strip())
    return out


def harvest(dirs, keep, arm='prob40'):
    """-> {tag: secs} for targets IN `keep` solved on `arm`."""
    got = {}
    for d in dirs:
        for f in glob.glob(os.path.join(F, d, 'logs', '*.out')):
            s = open(f, errors='ignore').read()
            m = RE.search(s)
            if not m:
                continue
            p = OUT.search(s)
            if not p or f'/{arm}/' not in p.group(1):
                continue
            tag = os.path.basename(p.group(1))[:-4]
            if tag not in keep:          # foreign target from a mixed tree
                continue
            got.setdefault(tag, float(m.group(1)))
    return got


def main():
    t100 = members('p101_test100_jobs.txt')
    t25 = members('p101_test25_jobs.txt')
    print(f'target lists: test100={len(t100)}  test25={len(t25)}  '
          f'overlap={len(t100 & t25)}\n')

    base = {'test100': harvest(['beamexp100'], t100),
            'test25': harvest(['beamexp25', 'beamexp'], t25)}
    cur = {'test100': harvest(['p101beam100'], t100),
           'test25': harvest(['p101beam25'], t25)}

    # elapsed for the RUNNING campaign
    start = None
    for d in ('p101beam100', 'p101beam25'):
        for f in glob.glob(os.path.join(F, d, 'logs', '*.out')):
            c = os.path.getctime(f)
            start = c if start is None else min(start, c)
    el = time.time() - start if start else 0
    print(f'p=101 elapsed: {el/60:.0f} min\n')

    print(f'{"set":>9} {"p101":>9} {"baseline @same":>15} {"baseline final":>15}')
    for k, tot in (('test100', 100), ('test25', 25)):
        b = base[k]
        n_at = sum(1 for s in b.values() if s <= el)
        print(f'{k:>9} {f"{len(cur[k])}/{tot}":>9} {f"{n_at}/{tot}":>15} '
              f'{f"{len(b)}/{tot}":>15}')
    a = sum(len(cur[k]) for k in cur)
    b = sum(sum(1 for s in base[k].values() if s <= el) for k in base)
    bf = sum(len(base[k]) for k in base)
    print(f'\n  COMBINED   p101 {a}/125   baseline@{el/60:.0f}min {b}/125   '
          f'baseline final {bf}/125')

    for k in ('test100', 'test25'):
        for nm, d in (('p101', cur[k]), ('base', base[k])):
            v = sorted(d.values())
            if v:
                print(f'  {k:8s} {nm}: n={len(v):3d} median {v[len(v)//2]:8.0f}s '
                      f'p90 {v[int(.9*(len(v)-1))]:9.0f}s max {v[-1]:9.0f}s '
                      f'total {sum(v)/3600:7.1f} CPU-h')
    paired()
    return 0




def paired():
    """PAIRED comparison on targets BOTH solved.

    Distribution summaries over DIFFERENT target sets are not a speed
    comparison: p101's test25 median covers only the 21 it has solved (the
    easiest 21), while the baseline's covers all 25. That difference alone
    produced a bogus '6x faster median' claim. Compare the same targets.
    """
    t100 = members('p101_test100_jobs.txt')
    t25 = members('p101_test25_jobs.txt')
    for name, keep, bdirs, cdir in (
            ('test100', t100, ['beamexp100'], 'p101beam100'),
            ('test25', t25, ['beamexp25', 'beamexp'], 'p101beam25')):
        b = harvest(bdirs, keep)
        c = harvest([cdir], keep)
        both = sorted(set(b) & set(c))
        if not both:
            continue
        r = sorted(c[t] / b[t] for t in both)
        cs = sum(c[t] for t in both)
        bs = sum(b[t] for t in both)
        faster = sum(1 for t in both if c[t] < b[t])
        print(f'\n{name}: {len(both)} targets solved by BOTH')
        print(f'  per-target ratio p101/base : median {r[len(r)//2]:.2f}x  '
              f'min {r[0]:.2f}x  max {r[-1]:.2f}x')
        print(f'  p101 faster on             : {faster}/{len(both)} '
              f'({100*faster/len(both):.0f}%)')
        print(f'  CPU-h on those targets     : p101 {cs/3600:.1f}  '
              f'base {bs/3600:.1f}  ({cs/bs:.2f}x)')


if __name__ == '__main__':
    sys.exit(main())
