#!/usr/bin/env python
"""Enumerate EVERY handle that can change the greedy worker's behavior.

Why a script and not a list I wrote by hand: I already got this wrong twice in
one sitting. A grep for single-quoted names missed the double-quoted ones. A
grep of greedy_reduce.py alone missed SAILIR_PACKED_RS, whose reader is in
greedy_worker.py (`greedy._PACKED_RS = ...`, poking the module constant from
outside), and missed SAILIR_STRIP_RAWS, whose reader is under sailir/. Any
by-hand list is a list of the handles I happened to remember.

WHAT COUNTS AS A HANDLE
  1. env  -- any SAILIR_*/OMP_*/MKL_*/MALLOC_* name read anywhere in the
             import graph the worker actually pulls in
  2. arg  -- any keyword parameter of greedy_reduce()
  3. cli  -- any flag greedy_worker.py's parser defines, plus --v7-cpus, which
             is not a parser flag at all: it is peeked out of sys.argv before
             import to size the CPU pin, and defaults to 8

Each handle is then classified against the certified record:
  PINNED   the record fixes it and something aborts if it differs
  FREE     deliberately variable (per-target paths, per-campaign limits)
  UNKNOWN  neither -- i.e. a way to change behavior that nobody is checking.
           These are the ones that matter. Every UNKNOWN is a hole.

Usage: audit_greedy_handles.py [--record reduction/greedy_certified.json]
"""
import argparse
import ast
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_RE = re.compile(
    r'''os\.environ(?:\.get|\.setdefault|\.pop)?\(\s*["']([A-Za-z_0-9]+)["']'''
    r'''|os\.environ\[\s*["']([A-Za-z_0-9]+)["']'''
    r'''|["']([A-Z_0-9]+)["']\s*in\s+os\.environ''')
INTERESTING = re.compile(r'^(SAILIR_|OMP_|MKL_|MALLOC_|OPENBLAS_|NUMEXPR_|PYTHON)')

# Files the greedy worker actually pulls in. Walking real imports would need to
# execute them (they abort without the guard env), so scan the two entry points
# plus the package they import from.
# Scope is derived, not chosen: start from the two entry points and follow
# `import X` to any reduction/X.py, transitively. Hand-picking the roots is how
# SAILIR_CANONICALIZE and SAILIR_CANON_MAPS_PKL stayed invisible -- they live in
# topo_config.py, which the worker imports but which was not in the list.
ROOTS = ['sailir']


def _closure():
    seen, todo = set(), ['reduction/greedy_worker.py', 'reduction/greedy_reduce.py']
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        try:
            src = open(os.path.join(REPO, rel), errors='ignore').read()
        except OSError:
            continue
        for m in re.findall(r'^\s*(?:from|import)\s+([a-z_][a-z_0-9]*)',
                            src, re.M):
            cand = f'reduction/{m}.py'
            if os.path.exists(os.path.join(REPO, cand)):
                todo.append(cand)
    return sorted(seen)


ROOTS = _closure() + ROOTS


def py_files():
    for r in ROOTS:
        p = os.path.join(REPO, r)
        if os.path.isfile(p):
            yield p
        else:
            for d, _, fs in os.walk(p):
                if '__pycache__' in d:
                    continue
                for f in fs:
                    if f.endswith('.py'):
                        yield os.path.join(d, f)


def env_handles():
    """name -> sorted list of files that read it."""
    out = {}
    for p in py_files():
        try:
            src = open(p, errors='ignore').read()
        except OSError:
            continue
        # drop comment-only lines so a mention in prose is not a read
        code = '\n'.join(l for l in src.splitlines()
                         if not l.lstrip().startswith('#'))
        for m in ENV_RE.finditer(code):
            name = m.group(1) or m.group(2) or m.group(3)
            if name and INTERESTING.match(name):
                out.setdefault(name, set()).add(os.path.relpath(p, REPO))
    return {k: sorted(v) for k, v in sorted(out.items())}


def arg_handles(path):
    tree = ast.parse(open(path).read())
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == 'greedy_reduce':
            a = n.args
            names = [x.arg for x in a.args][-len(a.defaults):] if a.defaults else []
            return names + [x.arg for x in a.kwonlyargs]
    return []


def cli_handles(path):
    out = set()
    for n in ast.walk(ast.parse(open(path).read())):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'add_argument'):
            for x in n.args:
                if isinstance(x, ast.Constant) and isinstance(x.value, str) \
                        and x.value.startswith('-'):
                    out.add(x.value)
    if '_peek_v7_cpus' in open(path).read():
        out.add('--v7-cpus')          # peeked from sys.argv, never parsed
    return sorted(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--record', default='reduction/greedy_certified.json')
    p.add_argument('--strict', action='store_true',
                   help='exit 1 if any handle is UNKNOWN')
    a = p.parse_args()

    rec_path = os.path.join(REPO, a.record)
    rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
    def keys(sec):        # value-maps carry a _why note; drop it
        return {k for k in rec.get(sec, {}) if not k.startswith('_')}

    def names(sec):       # name-lists
        return set(rec.get(sec, {}).get('names', []))

    pinned_env = keys('env_set') | keys('env_set_perf') | keys('env_model')
    # must-be-unset and derived count as ACCOUNTED FOR: the audit asks whether
    # any handle is UNEXAMINED, not whether every handle is pinned.
    free_env = (names('env_must_be_unset') | names('env_free')
                | names('env_derived'))
    pinned_arg = keys('params') | keys('params_fixed_by_worker')
    free_arg = names('params_free')
    pinned_cli = keys('cli')
    free_cli = names('cli_free') | names('cli_must_not_be_used')

    rc, unknown = 0, []

    print(f'RECORD: {a.record}' + ('' if rec else '   (MISSING -- nothing is pinned)'))

    envs = env_handles()
    print(f'\n=== ENV HANDLES ({len(envs)}) ===')
    for name, files in envs.items():
        cls = ('PINNED' if name in pinned_env else
               'FREE' if name in free_env else 'UNKNOWN')
        if cls == 'UNKNOWN':
            unknown.append(('env', name))
        where = files[0] + (f' +{len(files)-1}' if len(files) > 1 else '')
        print(f'  [{cls:7s}] {name:28s} {where}')

    args_ = arg_handles(os.path.join(REPO, 'reduction/greedy_reduce.py'))
    print(f'\n=== greedy_reduce() PARAMETERS ({len(args_)}) ===')
    for n in args_:
        cls = ('PINNED' if n in pinned_arg else
               'FREE' if n in free_arg else 'UNKNOWN')
        if cls == 'UNKNOWN':
            unknown.append(('arg', n))
        print(f'  [{cls:7s}] {n}')

    clis = cli_handles(os.path.join(REPO, 'reduction/greedy_worker.py'))
    print(f'\n=== greedy_worker.py CLI ({len(clis)}) ===')
    for n in clis:
        cls = ('PINNED' if n in pinned_cli else
               'FREE' if n in free_cli else 'UNKNOWN')
        if cls == 'UNKNOWN':
            unknown.append(('cli', n))
        print(f'  [{cls:7s}] {n}')

    must_unset = names('env_must_be_unset')
    leaked = sorted(k for k in must_unset if k in os.environ)
    print(f'\n=== MUST-BE-UNSET ({len(must_unset)}) ===')
    print('  unset in the certified environment; setting any one runs a config'
          '\n  nobody validated -- SAILIR_SYM_FIRST=1 skips the search entirely')
    print(f'  set in THIS shell: {leaked if leaked else "none"}')

    print(f'\n=== SUMMARY ===')
    print(f'  handles total : {len(envs) + len(args_) + len(clis)}')
    print(f'  UNKNOWN       : {len(unknown)}')
    for kind, n in unknown:
        print(f'      {kind}: {n}')
    if unknown:
        print('\n  Every UNKNOWN is a way to change worker behavior that nothing'
              '\n  checks. Classify each in the record as pinned or free.')
        rc = 1 if a.strict else 0
    else:
        print('  every handle is accounted for')
    return rc


if __name__ == '__main__':
    sys.exit(main())
