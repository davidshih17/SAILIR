#!/usr/bin/env python
"""Hash the DATA inputs the greedy worker reads, and check them against the record.

The last hole the handle audit cannot reach. Env vars, arguments and flags are
all now pinned in reduction/greedy_certified.json, but four inputs decide
behavior through their CONTENT, and the record pins at most their path:

  model checkpoint        pinned by path suffix only
  topology_input/<topo>/  not pinned
  canonical sectors pkl   not pinned
  transforms store pkl    not pinned

Edit any of them and every guard still passes while the answers change. That is
strictly worse than a wrong env var, which at least aborts. Note the overlap:
SAILIR_CANON_MAPS_PKL is banned from pointing at a different sector pickle, but
nothing stopped anyone from editing the pickle it does point at.

  --write   compute hashes and store them in the record
  --check   recompute and compare; exit 1 on any mismatch

Directories hash as the sorted list of (relative path, sha256) over their files,
so a renamed or added file changes the hash too. Large files are streamed rather
than read whole -- a checkpoint is hundreds of megabytes.
"""
import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(REPO, 'reduction', 'greedy_certified.json')


def sha_file(path, _buf=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(_buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha_dir(path):
    h = hashlib.sha256()
    for d, dirs, fs in os.walk(path):
        dirs.sort()
        for f in sorted(fs):
            if f.endswith(('.pyc', '.log')) or '__pycache__' in d:
                continue
            p = os.path.join(d, f)
            h.update(os.path.relpath(p, path).encode())
            h.update(sha_file(p).encode())
    return h.hexdigest()


def targets(rec):
    """The data inputs, resolved from the record and topo_config."""
    out = {}
    ck = rec.get('cli', {}).get('checkpoint_tail')
    if ck:
        out['checkpoint'] = os.path.join(REPO, ck)
    # NOT the topology directory: it is 84 GB, and ~99.9% of that is Kira
    # OUTPUT (kira_nosym_backsub alone is 75 GB) which the worker never reads.
    # Hashing the tree would be slow AND wrong -- it would trip on generated
    # artifacts while telling us nothing about the inputs. Topology.from_dir
    # reads exactly two files; those are the topology.
    topo = rec.get('env_model', {}).get('SAILIR_TOPOLOGY')
    if topo:
        d = os.path.join(REPO, 'topology_input', topo)
        for f in ('integralfamilies.yaml', 'kinematics.yaml'):
            if os.path.exists(os.path.join(d, f)):
                out[f'topology/{f}'] = os.path.join(d, f)
    # the symmetry transform store, when the record pins one (p101 routing)
    sym = rec.get('env_model', {}).get('SAILIR_SYM_STORE')
    if sym:
        out['sym_store_p101'] = os.path.join(REPO, sym)

    sys.path.insert(0, os.path.join(REPO, 'reduction'))
    os.environ.setdefault('SAILIR_TOPOLOGY', topo or 'gravity3L')
    # STRIP THE PATH OVERRIDES FIRST. topo_config.STORE_PKL and CANON_PKL are
    # env-overridable (SAILIR_SYM_STORE / SAILIR_CANON_MAPS_PKL). The certified
    # runner exports SAILIR_SYM_STORE=<p101 store>, so resolving through
    # topo_config here hashed the p101 file and compared it against the 1009
    # entry -- reporting the production store as CORRUPT when it was untouched.
    # The p101 store has its own record entry (sym_store_p101) keyed by explicit
    # path, so these two must resolve to the topology's OWN defaults.
    _saved = {k: os.environ.pop(k)
              for k in ('SAILIR_SYM_STORE', 'SAILIR_CANON_MAPS_PKL')
              if k in os.environ}
    sys.modules.pop('topo_config', None)     # force a re-read of the defaults
    try:
        import topo_config as tc
        for name in ('CANON_PKL', 'STORE_PKL'):
            p = getattr(tc, name, None)
            if p:
                out[name.lower()] = p
    except Exception as e:                       # noqa: BLE001
        print(f'  note: topo_config not importable ({e}); '
              f'hashing only checkpoint + topology', file=sys.stderr)
    finally:
        os.environ.update(_saved)
        sys.modules.pop('topo_config', None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--write', action='store_true')
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    if not (a.write or a.check):
        p.error('pick --write or --check')

    rec = json.load(open(RECORD))
    tg = targets(rec)

    now = {}
    for name, path in sorted(tg.items()):
        if not os.path.exists(path):
            print(f'  MISSING {name}: {path}', file=sys.stderr)
            continue
        now[name] = {
            'path': os.path.relpath(path, REPO),
            'sha256': sha_dir(path) if os.path.isdir(path) else sha_file(path),
        }

    if a.write:
        rec['data_inputs'] = dict(
            _why=('CONTENT hashes of the files the certified run read. Paths '
                  'alone are not enough: editing the pickle a pinned path '
                  'points at changes the answers while every other guard '
                  'passes. Directories hash as sorted (relpath, sha256) over '
                  'their files, so an added or renamed file also changes it. '
                  'Verify with scripts/greedy_data_hashes.py --check.'),
            **now)
        json.dump(rec, open(RECORD, 'w'), indent=2)
        for k, v in now.items():
            print(f'  {k:14s} {v["sha256"][:16]}...  {v["path"]}')
        print(f'\nwrote {len(now)} hashes to {os.path.relpath(RECORD, REPO)}')
        return 0

    have = {k: v for k, v in rec.get('data_inputs', {}).items()
            if not k.startswith('_')}
    if not have:
        print('record has no data_inputs section; run --write first')
        return 1
    bad = 0
    for k, v in sorted(have.items()):
        cur = now.get(k, {}).get('sha256')
        ok = cur == v['sha256']
        print(f'  [{"ok  " if ok else "DIFF"}] {k:14s} {v["path"]}')
        if not ok:
            bad += 1
            print(f'         certified {v["sha256"][:16]}...')
            print(f'         now       {(cur or "MISSING")[:16]}...')
    print(f'\n{len(have)-bad}/{len(have)} data inputs match the certified run')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
