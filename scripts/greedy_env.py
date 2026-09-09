#!/usr/bin/env python
"""Emit the certified greedy environment as shell, straight from the record.

Lets a shell wrapper stop hand-copying the configuration:

    eval "$(python scripts/greedy_env.py --export)"

Why this exists: reduction/greedy_certified.json became the single source of
truth for greedy_reduce.py and hierarchical_reduction.py, but the wrapper that
the certified 125 actually run through -- results/truth/finetune/greedy_worker.sh
-- kept its own hardcoded copy of the env, the prime, the cpus and the
checkpoint. Two callers, one of them not reading the record, is the same
silent-drift bug the record was written to remove, just moved to a shell script.

--export   export lines for env_set, env_set_perf and env_model
--unset    unset lines for every must-be-unset name, so a variable inherited
           from the submitting shell cannot leak into the job. Without this the
           record only *describes* the unset half of the configuration.
--cli      the certified prime / v7-cpus / checkpoint as shell variables, so a
           wrapper can build its command line from the record too
--check    verify the CURRENT environment matches; exit 1 if not
"""
import argparse
import json
import os
import shlex
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(REPO, 'reduction', 'greedy_certified.json')


def load():
    with open(RECORD) as f:
        return json.load(f)


def values(rec, *secs):
    return {k: v for sec in secs
            for k, v in rec.get(sec, {}).items() if not k.startswith('_')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--export', action='store_true')
    p.add_argument('--unset', action='store_true')
    p.add_argument('--cli', action='store_true')
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    rec = load()

    env = values(rec, 'env_set', 'env_set_perf', 'env_model')
    must_unset = rec.get('env_must_be_unset', {}).get('names', [])
    cli = values(rec, 'cli')

    if not any((a.export, a.unset, a.cli, a.check)):
        p.error('pick one of --export/--unset/--cli/--check')

    if a.export:
        print(f'# from {os.path.relpath(RECORD, REPO)} -- do not hand-edit')
        for k, v in sorted(env.items()):
            print(f'export {k}={shlex.quote(str(v))}')
    if a.unset:
        for k in sorted(must_unset):
            print(f'unset {k}')
    if a.cli:
        print(f'GREEDY_PRIME={cli["prime"]}')
        print(f'GREEDY_V7_CPUS={cli["v7_cpus"]}')
        print(f'GREEDY_CKPT={shlex.quote(cli["checkpoint_tail"])}')
    if a.check:
        bad = {k: os.environ.get(k) for k, v in env.items()
               if os.environ.get(k) != str(v)}
        leaked = {k: os.environ[k] for k in must_unset if k in os.environ}
        for k, v in sorted(bad.items()):
            print(f'MISMATCH {k}={v!r} (certified {env[k]!r})', file=sys.stderr)
        for k, v in sorted(leaked.items()):
            print(f'LEAKED   {k}={v!r} (certified: UNSET)', file=sys.stderr)
        if bad or leaked:
            return 1
        print(f'environment matches the certified record '
              f'({len(env)} set, {len(must_unset)} unset)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
