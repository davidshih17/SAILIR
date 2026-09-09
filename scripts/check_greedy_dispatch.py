#!/usr/bin/env python
"""Verify the orchestrator's greedy dispatch, and that it disturbs nothing else.

Three things must hold, and only the third is about greedy at all:

1. DEFAULT PATH UNCHANGED. The v7/v9 submit file must be byte-identical to the
   one the committed orchestrator produced. The greedy branch is additive and
   default-off; if adding it moved so much as a space in the ordinary path,
   every existing run changes.

2. GREEDY CLI ACCEPTED. Every flag the orchestrator emits for greedy must be a
   flag greedy_worker.py's parser actually defines. This is the check that was
   missing when a stripped build killed 125 jobs at once: the caller and the
   callee were edited in separate steps and nothing compared them. The v7/v9
   line would fail here three times over -- --beam_width, --v7-cpus and the
   --no-tabu that rides in on SAILIR_WORKER_EXTRA_FLAGS.

3. GUARD ENV SATISFIED. greedy_reduce refuses to start unless all eight
   pinned variables hold, and an UNSET one fails too, so the submit's
   environment line must carry every one of them -- with each name assigned
   exactly once.

Usage:  check_greedy_dispatch.py [--ref-rev HEAD]
"""
import argparse
import ast
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORCH = os.path.join(REPO, 'reduction', 'hierarchical_reduction.py')
WORKER = os.path.join(REPO, 'reduction', 'greedy_worker.py')

CALL = dict(
    integral=(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0),
    job_name='j0',
    # must be the CERTIFIED checkpoint: greedy dispatch refuses any other
    model_checkpoint='/repo/checkpoints/gravity3L_p101_scratch/best_model.pt',
    beam_width=1, max_steps=1000000, prime=101,
    topology_dir='/topo/gravity3L',
    paper_masters_only=True, use_v7_worker=True, v7_cpus=1, memory_gb=4,
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def submit_text(mod, work_dir, env):
    """Render one submit file under a given environment."""
    old = dict(os.environ)
    try:
        for k in ('SAILIR_WORKER_GREEDY', 'SAILIR_WORKER_V9', 'SAILIR_SCORE',
                  'SAILIR_NM_PENALTY', 'SAILIR_BEAM_TOTAL', 'SAILIR_BEAM_SORT',
                  'SAILIR_SECTOR_RANK', 'SAILIR_TOPOLOGY', 'SAILIR_SYM_DROP',
                  'SAILIR_SUCCESS_TOTAL', 'SAILIR_PACKED_RS',
                  'SAILIR_STRIP_RAWS', 'SAILIR_V9_UPENUM', 'SAILIR_V9_CULL'):
            os.environ.pop(k, None)
        os.environ.update(env)
        os.makedirs(os.path.join(work_dir, 'logs'), exist_ok=True)
        p = mod.create_condor_submit(
            work_dir=Path(work_dir),
            output_file=os.path.join(work_dir, 'r.pkl'), **CALL)
        return open(p).read()
    finally:
        os.environ.clear()
        os.environ.update(old)


def worker_flags(path):
    """Flags greedy_worker.py's parser defines, read from the AST."""
    out = set()
    for n in ast.walk(ast.parse(open(path).read())):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'add_argument'):
            for a in n.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    out.add(a.value)
    return out


def emitted_flags(text):
    m = re.search(r'^arguments\s*=(.*)$', text, re.M)
    line = m.group(1) if m else ''
    return [t for t in re.findall(r'(?<![\w=])(--?[A-Za-z][\w-]*)', line)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref-rev', default='HEAD')
    a = p.parse_args()
    rc = 0

    mod = load(ORCH, 'orch_new')

    # ---- 1. default path byte-identical to the committed orchestrator -------
    ref_src = subprocess.run(
        ['git', '-C', REPO, 'show', f'{a.ref_rev}:reduction/hierarchical_reduction.py'],
        capture_output=True, text=True).stdout
    with tempfile.TemporaryDirectory() as td:
        rp = os.path.join(td, 'ref_orch.py')
        open(rp, 'w').write(ref_src)
        ref_mod = load(rp, 'orch_ref')
        base_env = {'SAILIR_WORKER_V9': '1', 'SAILIR_SCORE': 'local',
                    'SAILIR_NM_PENALTY': '0', 'SAILIR_BEAM_TOTAL': '1',
                    'SAILIR_BEAM_SORT': 'prob', 'SAILIR_SECTOR_RANK': '1',
                    'SAILIR_TOPOLOGY': 'gravity3L'}
        old_t = submit_text(ref_mod, os.path.join(td, 'a'), base_env)
        new_t = submit_text(mod, os.path.join(td, 'b'), base_env)
        # The reference module is exec'd from a temp path, so ITS REPO_DIR
        # resolves there. Normalize both repo roots, else every run reports a
        # false CHANGED on the worker path alone.
        old_n = old_t.replace(os.path.join(td, 'a'), 'WD').replace(
            os.path.dirname(os.path.dirname(rp)), 'REPO').replace(td, 'REPO')
        new_n = new_t.replace(os.path.join(td, 'b'), 'WD').replace(REPO, 'REPO')
        ok = old_n == new_n
        print(f'[1] default v7/v9 path unchanged vs {a.ref_rev}: '
              f'{"IDENTICAL" if ok else "CHANGED"}')
        if not ok:
            rc = 1
            for l1, l2 in zip(old_n.splitlines(), new_n.splitlines()):
                if l1 != l2:
                    print(f'      was: {l1}\n      now: {l2}')

        # ---- 2 + 3. greedy path -------------------------------------------
        g = submit_text(mod, os.path.join(td, 'c'), {'SAILIR_WORKER_GREEDY': '1',
                                                     'SAILIR_TOPOLOGY': 'gravity3L'})

    script = re.search(r'arguments\s*=\s*-u\s+\S*/(\w+\.py)', g)
    print(f'[2] worker script            : {script.group(1) if script else "??"}')
    if not script or script.group(1) != 'greedy_worker.py':
        print('    FAIL: greedy not selected'); rc = 1

    wsrc = open(WORKER).read()
    defined = worker_flags(WORKER)
    emitted = [f for f in emitted_flags(g) if f != '-u']   # python's own flag
    # --v7-cpus is NOT an argparse flag: it is peeked straight out of sys.argv
    # before greedy_reduce is imported, to size the thread cap and CPU pin.
    # Absent, it defaults to 8 -> an 8-core pin in a 1-CPU slot.
    peeked = {'--v7-cpus'} if '_peek_v7_cpus' in wsrc else set()
    unknown = [f for f in emitted if f not in defined and f not in peeked]
    print(f'[2] flags emitted            : {" ".join(emitted)}')
    if unknown:
        # harmless ONLY because the parser tolerates unknowns -- say which
        tol = 'parse_known_args' in wsrc
        print(f'    {"WARN" if tol else "FAIL"}: not defined by the parser: '
              f'{" ".join(unknown)}'
              f'{"  (ignored: parse_known_args)" if tol else ""}')
        rc = rc if tol else 1
    else:
        print(f'    all {len(emitted)} accounted for '
              f'({len(peeked & set(emitted))} peeked from sys.argv)')
    if '--v7-cpus' not in emitted:
        print('    FAIL: --v7-cpus missing -> worker defaults to 8 cores'); rc = 1

    envm = re.search(r'^environment\s*=\s*"(.*)"$', g, re.M)
    env = dict(kv.split('=', 1) for kv in (envm.group(1).split() if envm else []))
    names = [kv.split('=', 1)[0] for kv in (envm.group(1).split() if envm else [])]
    need = mod._GREEDY_ENV
    missing = {k: v for k, v in need.items() if env.get(k) != v}
    dup = {n for n in names if names.count(n) > 1}
    print(f'[3] guard env ({len(need)} pinned)    : '
          f'{"all present and correct" if not missing else "INCOMPLETE"}')
    for k, v in sorted(missing.items()):
        print(f'    FAIL {k}: submit has {env.get(k)!r}, guard needs {v!r}')
    if missing:
        rc = 1
    if dup:
        print(f'    FAIL: assigned twice: {" ".join(sorted(dup))}'); rc = 1

    # [5] the CLI values the env guard cannot see. A wrong prime is SILENT:
    # right answers-shaped output, wrong coefficients, every other check green.
    CK = 'checkpoints/gravity3L_p101_scratch/best_model.pt'
    cases = [('prime 1009 (the orchestrator DEFAULT)', dict(prime=1009, v7_cpus=1, model_checkpoint=CK)),
             ('v7_cpus 8',                             dict(prime=101, v7_cpus=8, model_checkpoint=CK)),
             ('a different checkpoint',                dict(prime=101, v7_cpus=1, model_checkpoint='ck/other.pt')),
             ('the certified triple',                  dict(prime=101, v7_cpus=1, model_checkpoint='/x/'+CK))]
    for label, kw in cases:
        want_ok = label.startswith('the certified')
        try:
            mod.greedy_check_certified(**kw); got_ok = True
        except SystemExit:
            got_ok = False
        ok = (got_ok == want_ok)
        print(f'[5] {label:38s}: '
              f'{"accepted" if got_ok else "rejected"}  {"" if ok else "<-- WRONG"}')
        if not ok:
            rc = 1

    # conflicting orchestrator env must be a LOUD error, not a silent override
    old = dict(os.environ)
    try:
        os.environ['SAILIR_WORKER_GREEDY'] = '1'
        os.environ['SAILIR_NM_PENALTY'] = '0'      # the g1023 runner's value
        try:
            mod.greedy_env_string()
            print('[4] conflicting NM_PENALTY=0 : FAIL (silently accepted)'); rc = 1
        except SystemExit:
            print('[4] conflicting NM_PENALTY=0 : rejected loudly')
    finally:
        os.environ.clear(); os.environ.update(old)

    print('\n' + ('OK' if rc == 0 else 'FAIL'))
    return rc


if __name__ == '__main__':
    sys.exit(main())
