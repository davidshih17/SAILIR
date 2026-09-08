"""Map CURRENT-build line numbers back to ORIGINAL beam_search_v9.py lines.

Every spec is written in ORIGINAL coordinates so stages compose, but analysis
(dead flags, stale comments) runs on the CURRENT build. This replays the
emitter's kill set to produce: current_line -> original_line.
"""
import ast, json, sys
S='/tmp/claude-706/-het-p4-dshih/422e8a7a-68ad-4e51-a91e-bd72ad2e933b/scratchpad'
spec=json.load(open(sys.argv[1]))
hits=set(json.load(open(f'{S}/hits_short.json')))|set(json.load(open(f'{S}/hits_med.json')))
src=open('reduction/beam_search_v9.py').read().splitlines(keepends=True)
tree=ast.parse(''.join(src))
kill=set()
for n in tree.body:
    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
        if not (set(range(n.body[0].lineno,(n.end_lineno or n.lineno)+1)) & hits):
            kill.update(range(n.lineno,(n.end_lineno or n.lineno)+1))
for a,b in spec.get('delete',[]): kill.update(range(a,b+1))
rep={int(r['line']) for r in spec.get('replace',[])}
out=[]
for i in range(1,len(src)+1):
    if i in kill: continue
    # a multi-line replacement occupies >1 current line; record the original for each
    n=1
    for r in spec.get('replace',[]):
        if int(r['line'])==i: n=r['text'].count('\n')+1
    out += [i]*n
json.dump(out, open(sys.argv[2],'w'))
print(f"map: {len(out)} current lines -> original")
