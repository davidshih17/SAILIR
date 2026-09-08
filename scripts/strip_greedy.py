#!/usr/bin/env python
"""Emit a stripped greedy build from beam_search_v9.py.

Every surviving line is copied VERBATIM; the only edit ever applied is removing
4 leading spaces from a promoted else-body. Nothing is authored.

Two removal rules:
  * stage 1  -- top-level functions the trace never entered (automatic)
  * batches  -- explicit spans, each a WHOLE branch at its decision point

DELETE spans must be whole branches. Deleting a FRAGMENT of a branch is how the
`_blk` break happened: `if _tset: ... else: _blk = []` was cut out of the middle
of a dead block, leaving `_n_blocked = len(_blk)` below it reading a name that
no longer existed. It survived 125/125 because --no-tabu made it unreachable.
Hence check_dangling.py, which is not optional.

PROMOTE spans are live else-bodies that must survive the removal of their `if`.

Spec JSON:
  {"delete": [[a,b],...], "promote": [[a,b],...], "out": "reduction/x.py"}

All line numbers are in ORIGINAL beam_search_v9.py coordinates, so batches
compose without remapping and any build is reproducible from the original.
"""
import argparse
import ast
import json
import os
import sys

SRC = 'reduction/beam_search_v9.py'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('spec', help='JSON with delete/promote/out')
    p.add_argument('--hits', nargs='+', required=True,
                   help='trace hit-line JSON files (union is the live set)')
    p.add_argument('--src', default=SRC)
    a = p.parse_args()

    spec = json.load(open(a.spec))
    hits = set()
    for h in a.hits:
        hits |= set(json.load(open(h)))

    src = open(a.src).read().splitlines(keepends=True)
    tree = ast.parse(''.join(src))

    kill = set()
    n_stage1 = 0
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = range(n.body[0].lineno, (n.end_lineno or n.lineno) + 1)
            if not (set(body) & hits):
                kill.update(range(n.lineno, (n.end_lineno or n.lineno) + 1))
                n_stage1 += 1

    n_batch = 0
    for lo, hi in spec.get('delete', []):
        before = len(kill)
        kill.update(range(lo, hi + 1))
        n_batch += len(kill) - before

    # A line may sit inside SEVERAL nested folds -- e.g. the _RESERVE_PARENT
    # else-body is itself inside the beam_sort=='prob' else-body. Each enclosing
    # fold removes one indent level, so count the ranges covering a line rather
    # than treating promote as a set (which silently under-dedents and yields
    # "unexpected indent").
    promote = {}
    for lo, hi in spec.get('promote', []):
        for i in range(lo, hi + 1):
            promote[i] = promote.get(i, 0) + 1

    # A promoted else-body may legitimately CONTAIN a deleted block (e.g. the
    # tabu branch nested inside the serial enumerate body). Deletion wins; the
    # surviving outer lines still dedent. Report it, don't reject it.
    overlap = set(promote) & kill
    if overlap:
        print(f'  note: {len(overlap)} promoted line(s) also deleted '
              f'(nested dead block inside a promoted body) -- deletion wins')

    # REPLACE: for a line that carries BOTH something to remove and something
    # to keep, e.g. `resume_from=None, tabu=False,` -- deletion would take the
    # survivor with it. The replacement text is given verbatim in the spec, so
    # the edit is still declared as data, not authored here.
    replace = {int(r['line']): r['text'] for r in spec.get('replace', [])}
    for r in spec.get('replace', []):
        ln, old = int(r['line']), src[int(r['line']) - 1].rstrip('\n')
        # `expect` pins the original text: line numbers are ORIGINAL-file
        # coordinates, so a stale spec would silently rewrite the wrong line.
        if 'expect' in r and r['expect'] != old:
            sys.exit(f'ERROR: line {ln} is {old!r}, spec expected {r["expect"]!r}')
        print(f'  replace {ln}: {old.strip()[:44]!r} -> {r["text"].strip()[:44]!r}')

    out = []
    for i in range(1, len(src) + 1):
        if i in kill:
            continue
        line = replace[i] + '\n' if i in replace else src[i - 1]
        n_ded = promote.get(i, 0)
        if n_ded and line.strip():
            if not line.startswith('    ' * n_ded):
                sys.exit(f'ERROR: line {i} needs {n_ded} dedent(s) but has too '
                         f'little indent: {line!r}')
            line = line[4 * n_ded:]
        out.append(line)

    dest = spec['out']
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    open(dest, 'w').write(''.join(out))

    try:
        ast.parse(''.join(out))
    except SyntaxError as e:
        sys.exit(f'ERROR: emitted file does not parse: line {e.lineno}: {e.msg}')

    print(f'{dest}: {len(out)} lines (from {len(src)}, removed {len(src)-len(out)})')
    print(f'  stage-1 functions removed : {n_stage1}')
    print(f'  batch lines deleted       : {n_batch}')
    print(f'  else-body lines promoted  : {len(promote)} '
          f'({sum(1 for v in promote.values() if v > 1)} nested, dedented twice)')
    print('  parses OK')


if __name__ == '__main__':
    main()
