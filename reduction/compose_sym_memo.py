#!/usr/bin/env python
"""ONE-TIME conversion of a persisted routing table to COMPOSED form.

Raw rules rewrite an integral into terms that may themselves be routable;
folding them cascades intermediates through the orchestrator's expression.
This composes every rule to a flat combination over survivors (terms with no
rule in the table are treated as survivors — if they later become routing
candidates the orchestrator routes them then; exactness is unaffected).
Pure dictionary arithmetic mod p; no routing solves.

Usage: compose_sym_memo.py <work_dir>   (backs up sym_memo.pkl first)
"""
import os
import sys
import pickle
import shutil
import time

P = 1009
work = sys.argv[1]
path = os.path.join(work, "sym_memo.pkl")
with open(path, "rb") as f:
    table = pickle.load(f)
n_rules = sum(1 for v in table.values() if v is not None)
print(f"loaded {len(table)} entries ({n_rules} rules) from {path}")

comp = {}
t0 = time.time()
done = 0
for top, rule in table.items():
    if rule is None or top in comp:
        continue
    stack = [top]
    while stack:
        j = stack[-1]
        if j in comp:
            stack.pop()
            continue
        r = table.get(j)
        deps = [k for k in r if k not in comp and table.get(k) is not None]
        if deps:
            stack.extend(deps)
            continue
        out = {}
        for k, c in r.items():
            sub = comp.get(k) if table.get(k) is not None else None
            if sub is None:
                out[k] = (out.get(k, 0) + c) % P
            else:
                for l, cl in sub.items():
                    v = (out.get(l, 0) + c * cl) % P
                    if v:
                        out[l] = v
                    else:
                        out.pop(l, None)
        comp[j] = {k: v for k, v in out.items() if v}
        stack.pop()
    done += 1
    if done % 20000 == 0:
        print(f"  ... {done}/{n_rules} composed, {time.time()-t0:.0f}s",
              flush=True)

new_table = {k: (comp[k] if v is not None else None)
             for k, v in table.items()}
sizes = sorted(len(v) for v in new_table.values() if v is not None)
print(f"composed {len(comp)} rules in {time.time()-t0:.0f}s; "
      f"rule sizes median={sizes[len(sizes)//2]} p95={sizes[int(len(sizes)*.95)]} "
      f"max={sizes[-1]}")
bak = path.replace(".pkl", "_raw_backup.pkl")
if not os.path.exists(bak):
    shutil.copy2(path, bak)
    print(f"backed up raw table -> {bak}")
with open(path + ".tmp", "wb") as f:
    pickle.dump(new_table, f)
os.replace(path + ".tmp", path)
print(f"installed composed table -> {path}")
