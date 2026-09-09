#!/usr/bin/env python
"""Build ONE Condor submit file through hierarchical_reduction.create_condor_submit.

The point is to exercise the real function rather than a hand-written copy of
what I think it emits, so the thing under test is the dispatch code path.

Prints SUBMIT_FILE=<path> for the caller to submit, plus the two lines that
decide whether the job can work at all: `arguments` (is --v7-cpus there? it
defaults to 8 and would pin 8 cores inside a 1-CPU slot) and `environment`
(are all eight guard variables present? greedy_reduce aborts at import if not,
and an UNSET one aborts too).
"""
import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work-dir', required=True)
    p.add_argument('--target', required=True, help='comma-separated integral')
    p.add_argument('--output', required=True)
    p.add_argument('--topology', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prime', type=int, required=True)
    p.add_argument('--max-steps', type=int, default=3000)
    a = p.parse_args()

    orch = load(REPO / 'reduction' / 'hierarchical_reduction.py', 'orch')

    if not orch.greedy_selected():
        sys.exit('SAILIR_WORKER_GREEDY is not 1 -- this would dispatch v7/v9')

    integral = tuple(int(x) for x in a.target.split(','))
    wd = Path(a.work_dir)
    (wd / 'logs').mkdir(parents=True, exist_ok=True)

    sub = orch.create_condor_submit(
        work_dir=wd,
        integral=integral,
        job_name='orchdisp',
        output_file=a.output,
        model_checkpoint=a.checkpoint,
        beam_width=1,
        max_steps=a.max_steps,
        prime=a.prime,
        topology_dir=a.topology,
        paper_masters_only=True,
        use_v7_worker=True,
        v7_cpus=1,
        memory_gb=4,
    )

    txt = open(sub).read()
    args_line = re.search(r'^arguments\s*=.*$', txt, re.M)
    env_line = re.search(r'^environment\s*=\s*"(.*)"$', txt, re.M)
    print(f'SUBMIT_FILE={sub}')
    print(f'\n{args_line.group(0) if args_line else "NO arguments LINE"}')
    print(f'\nenvironment:')
    env = dict(kv.split('=', 1) for kv in (env_line.group(1).split() if env_line else []))
    for k, v in sorted(env.items()):
        mark = ''
        if k in orch._GREEDY_ENV:
            mark = ' [guard]' if v == orch._GREEDY_ENV[k] else ' [GUARD MISMATCH]'
        print(f'  {k}={v}{mark}')
    miss = [k for k, v in orch._GREEDY_ENV.items() if env.get(k) != v]
    if miss:
        sys.exit(f'\nMISSING/WRONG guard vars -> worker will abort: {miss}')
    if '--v7-cpus' not in (args_line.group(0) if args_line else ''):
        sys.exit('\n--v7-cpus absent -> worker defaults to 8 cores in a 1-CPU slot')
    print('\nall guard vars present; --v7-cpus present')
    return 0


if __name__ == '__main__':
    sys.exit(main())
