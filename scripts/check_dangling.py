"""INVARIANT: no name assigned ONLY inside removed spans may still be read by
surviving code. This is the exact failure that shipped in batch 1 (_blk)."""
import ast, json, sys
def bound(tree, lines=None):
    """Every name a module can bind: assignment, import-as, with-as, except-as,
    for/comprehension target, function arg, def/class name, walrus."""
    out = set()
    for n in ast.walk(tree):
        ln = getattr(n, 'lineno', None)
        inr = (lines is None) or (ln in lines)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and inr:
            out.add(n.id)
        elif isinstance(n, ast.alias) and inr:
            out.add((n.asname or n.name).split('.')[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and inr:
            out.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name and inr:
            out.add(n.name)
        elif isinstance(n, ast.arg) and inr:
            out.add(n.arg)
    return out
orig, stripped = sys.argv[1], sys.argv[2]
spec = json.load(open(sys.argv[3]))
if isinstance(spec, dict):           # full spec: delete spans AND replaced lines
    killed = set()
    for a, b in spec.get('delete', []): killed.update(range(a, b + 1))
    # A REPLACE can remove an assignment just as surely as a delete -- that gap
    # let `beam` survive as a dangling read after 3d-4 rebound it to `state`.
    killed |= {int(r['line']) for r in spec.get('replace', [])}
else:
    killed = set()
    for a, b in spec: killed.update(range(a, b + 1))
o = ast.parse(open(orig).read()); s = ast.parse(open(stripped).read())
read_after = {n.id for n in ast.walk(s) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Load)}
dangling = sorted((bound(o, killed) & read_after) - bound(s))
if dangling:
    print(f"FAIL: {len(dangling)} name(s) read but no longer assigned: {dangling}")
    sys.exit(1)
print("OK: no dangling names")
