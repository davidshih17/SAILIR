"""Final accounting for the p=101 closure campaign.

Counts closures on disk, unique targets covered, and cross-references the
group_*.json manifests to say exactly which targets are missing and why. The
status lines carry {tag, ok, sec, rows, rss_mb}; a target with NO status line
was never attempted (its group died before reaching it), which is a different
failure from one that ran and returned ok:false.
"""
import glob
import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{D}/v4_p101_18k'


def main():
    # ---- what the manifests asked for -----------------------------------
    want = {}                      # tag -> group id
    gfiles = sorted(glob.glob(f'{D}/groups_18k/group_*.json'))
    for f in gfiles:
        gid = int(os.path.basename(f)[6:-5])
        try:
            g = json.load(open(f))
        except Exception:
            continue
        tg = g.get('targets', g if isinstance(g, list) else [])
        for t in tg:
            want['_'.join(str(x) for x in t)] = gid
    print(f'targets requested across {len(gfiles):,} '
          f'groups: {len(want):,}')

    # ---- what ran --------------------------------------------------------
    seen, okt, badt = set(), set(), set()
    secs = []
    rss = []
    for f in glob.glob(f'{OUT}/status_group_*.jsonl'):
        for line in open(f, errors='ignore'):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            tag = r.get('tag')
            if tag is None:
                continue
            seen.add(tag)
            (okt if r.get('ok') else badt).add(tag)
            if r.get('sec') is not None:
                secs.append(r['sec'])
            if r.get('rss_mb'):
                rss.append(r['rss_mb'])
    print(f'\nattempted (status line present): {len(seen):,}')
    print(f'  ok=true  : {len(okt):,}')
    print(f'  ok=false : {len(badt - okt):,}')
    missing = set(want) - seen
    print(f'NEVER attempted (group died first): {len(missing):,}')

    # ---- closures on disk ------------------------------------------------
    files = glob.glob(f'{OUT}/**/*.json', recursive=True)
    files = [f for f in files if os.path.basename(f).startswith('closure')
             or '/out/' in f or '/retry' in f]
    tg = set()
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if 'integral' in d:
            tg.add('_'.join(str(x) for x in d['integral']))
    print(f'\nclosure files on disk : {len(files):,}')
    print(f'unique targets on disk: {len(tg):,}   '
          f'({100*len(tg)/max(len(want),1):.1f}% of requested)')

    if secs:
        s = sorted(secs)
        print(f'\nper-target solve seconds: median {s[len(s)//2]:.2f}  '
              f'p90 {s[int(.9*(len(s)-1))]:.1f}  max {s[-1]:.0f}')
    if rss:
        r = sorted(rss)
        print(f'group peak RSS (MB)     : median {r[len(r)//2]:,}  '
              f'max {r[-1]:,}')

    # which groups are entirely absent
    gmiss = {}
    for t in missing:
        gmiss.setdefault(want[t], 0)
        gmiss[want[t]] += 1
    print(f'\ngroups with missing targets: {len(gmiss):,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
