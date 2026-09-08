"""FENCE: load-bearing blocks that must survive every strip.
Each was mis-classified as 'dead' by trace coverage alone; each is reachable in
the benchmark config and removing it changes behaviour on inputs we care about.
Checked by SEMANTICS (AST), not by grepping text that a comment could fake."""
import ast, sys
src = open(sys.argv[1]).read()
tree = ast.parse(src)
fails = []

# 1. the stall exit: `if not tasks:` whose body can reach a `break`
stall = [n for n in ast.walk(tree) if isinstance(n, ast.If)
         and isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not)
         and isinstance(n.test.operand, ast.Name) and n.test.operand.id == 'tasks']
if not stall:
    fails.append("`if not tasks:` (stalled-search detection) is GONE")
elif not any(isinstance(x, ast.Break) for s in stall for x in ast.walk(s)):
    fails.append("`if not tasks:` survives but its `break` is GONE -> a stalled "
                 "search would loop forever")

# 2. the orchestrator's straggler resume path
if not [n for n in ast.walk(tree) if isinstance(n, ast.If)
        and any(isinstance(x, ast.Name) and x.id == 'resume_from'
                for x in ast.walk(n.test))]:
    fails.append("resume_from block is GONE -> orchestrator --resume-from breaks silently")

# 3. the compiled-cull fallback
if not [n for n in ast.walk(tree) if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name) and n.test.left.id == 'scored'
        and any(isinstance(c, ast.Constant) and c.value is None for c in n.test.comparators)]:
    fails.append("`if scored is None:` fallback is GONE -> compiled-cull decline becomes a crash")

if fails:
    for f in fails: print("FENCE FAIL:", f)
    sys.exit(1)
print("OK: all 3 fenced blocks present")
