"""INVARIANT: every module attribute the WORKER reads must still exist.

The four existing gates check the module against itself. They cannot see
`bs7._ACTION_SELECT` in the worker -- a cross-file reference. Phase 5 deleted
that flag and all 125 jobs died at startup with AttributeError, which no
static check of the module alone would ever catch.
"""
import ast, sys
mod, worker = sys.argv[1], sys.argv[2]
m = ast.parse(open(mod).read())
have = {n.name for n in m.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
for n in m.body:
    if isinstance(n, ast.Assign):
        for t in n.targets:
            if isinstance(t, ast.Name): have.add(t.id)
    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name): have.add(n.target.id)
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names: have.add((a.asname or a.name).split('.')[0])
w = ast.parse(open(worker).read())
alias = {a.asname for n in ast.walk(w) if isinstance(n, ast.Import)
         for a in n.names if a.asname and 'greedy' in (a.name or '')}
alias |= {a.asname or a.name for n in ast.walk(w) if isinstance(n, ast.ImportFrom)
          and n.module and 'greedy' in n.module for a in n.names}
# A reference inside an `if` CRASHES only when that branch runs; an
# unconditional one crashes every startup (the banner did). Report them
# differently -- flagging guarded refs as fatal would block correct builds.
guarded = set()
for n in ast.walk(w):
    if isinstance(n, ast.If):
        for c in ast.walk(n):
            if isinstance(c, ast.Attribute) and isinstance(c.value, ast.Name) \
               and c.value.id in alias:
                guarded.add(c.attr)
refs = {}
for n in ast.walk(w):
    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias:
        refs.setdefault(n.attr, n.lineno)
missing = sorted((a, l) for a, l in refs.items() if a not in have)
allow = set(sys.argv[3].split(',')) if len(sys.argv) > 3 else set()
fatal = [(a, l) for a, l in missing if a not in allow]
warn  = [(a, l) for a, l in missing if a in allow]
print(f'worker reads {len(refs)} module attributes via {sorted(alias)}')
for a, l in warn:
    print(f'  allowed: {a} missing, explicitly whitelisted ({worker}:{l})')
if fatal:
    print(f'FAIL: {len(fatal)} worker reference(s) missing from the module:')
    for a, l in fatal: print(f'   {worker}:{l}  ->  {a}')
    sys.exit(1)
print('OK: every worker reference resolves (or is whitelisted)')
