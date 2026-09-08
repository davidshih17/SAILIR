"""Find every `if <DEAD FLAG>:` block, the tabu removal done systematically.

A flag is DEAD when the benchmark never sets it, so the branch cannot run.
Reports the whole If node (and its else-body, which must be PROMOTED, not
deleted) plus every assignment that reads the flag out of the environment.
"""
import ast, json, sys
SRC = 'reduction/greedy_p3g.py'
# flags whose gating env var the benchmark never sets -> branch is unreachable
DEAD = {
 '_v5_prof','_trace_fh','_TRUTH_TRACE','_truth_actions','_DAGGER_MODE','_DAGGER_OUT',
 '_DAGGER_MISSING','_DAGGER_DEADEND','_V9_ORACLE_PATH','_V9_ORACLE_RANK',
 '_V9_ORACLE_STRICT','_V9_ORACLE','_pick_dump_dir','_precap_dump_dir','_LIB','_LIB_TRACE',
 '_CLOSURE_PROBE','_CLOSURE_PROBE_SET','_ACTION_SCORE','_ACTION_SELECT','_SYM_DROP',
 '_MW2_PRICE','_STRIP_TOTAL','_memprobe_path','_mem_bd','_reg_probe','_dump_steps',
 '_beam_set_trace_path','_beam_set_history','_mmap_thr','_PROB_FLOOR','_TOPN_PROB',
 '_RANK_SCORE','_REL_SCORE','_W1_DROP','_W1_FIRST','_RESERVE_PARENT','_ENT_BONUS',
 '_ENT_CUT','_dedup_key','_dedup_key_top','_dump_step','_dump_stuck_path',
}
src = open(SRC).read(); lines = src.splitlines(); tree = ast.parse(src)
blocks, assigns = [], []
def names(n):
    return {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
for n in ast.walk(tree):
    if isinstance(n, ast.If):
        t = names(n.test)
        if t and t <= DEAD:                       # test uses ONLY dead flags
            oe = (n.orelse[0].lineno, n.orelse[-1].end_lineno) if n.orelse else None
            blocks.append(dict(a=n.lineno, b=(oe[0]-1 if oe else n.end_lineno),
                               promote=list(oe) if oe else None,
                               head=lines[n.lineno-1].strip()[:60]))
    if isinstance(n, ast.Assign) and len(n.targets) == 1 \
       and isinstance(n.targets[0], ast.Name) and n.targets[0].id in DEAD:
        assigns.append(dict(a=n.lineno, b=n.end_lineno,
                            head=lines[n.lineno-1].strip()[:60]))
blocks.sort(key=lambda d: -(d['b']-d['a']))
tot = sum(d['b']-d['a']+1 for d in blocks) + sum(d['b']-d['a']+1 for d in assigns)
print(f"{len(blocks)} dead-flag IF blocks + {len(assigns)} flag assignments = {tot} lines")
print("\nlargest blocks:")
for d in blocks[:14]:
    p = f"  PROMOTE {d['promote']}" if d['promote'] else ""
    print(f"  {d['a']:5d}-{d['b']:<5d} {d['b']-d['a']+1:4d}  {d['head']}{p}")
json.dump(dict(blocks=blocks, assigns=assigns), open(sys.argv[1], 'w'))
