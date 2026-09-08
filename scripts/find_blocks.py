"""Find MAXIMAL dead functional blocks inside LIVE functions.

Unit = a whole branch at its decision point, never a fragment:
  if   : body unhit -> drop (if..body), promote orelse   | orelse unhit -> drop orelse
  for/while : body unhit (loop never ran) -> drop the loop
  try  : a handler unhit -> drop that handler
Maximal only: once a block is taken, nothing inside it is considered separately.
"""
import ast, json
S='/tmp/claude-706/-het-p4-dshih/422e8a7a-68ad-4e51-a91e-bd72ad2e933b/scratchpad'
hits=set(json.load(open(f'{S}/hits_short.json')))|set(json.load(open(f'{S}/hits_med.json')))
src=open('reduction/beam_search_v9.py').read().splitlines()
tree=ast.parse('\n'.join(src))
def span(ns): return (ns[0].lineno, ns[-1].end_lineno)
def dead(ns):
    a,b=span(ns); return not (set(range(a,b+1)) & hits)
def real(a,b):
    return sum(1 for i in range(a,b+1)
               if src[i-1].strip() and not src[i-1].strip().startswith('#'))
found=[]
def walk(node):
    for n in ast.iter_child_nodes(node):
        got=False
        if isinstance(n,ast.If):
            if n.body and dead(n.body):
                # drop test+body; orelse (if any) is promoted by the emitter
                found.append(dict(kind='if-body',a=n.lineno,b=span(n.body)[1],
                    promote=list(span(n.orelse)) if n.orelse else None,
                    head=src[n.lineno-1].strip()[:64])); got=True
            elif n.orelse and dead(n.orelse):
                a,b=span(n.orelse)
                # the `else:`/`elif` keyword line sits just above a's block
                found.append(dict(kind='orelse',a=a if src[a-1].lstrip().startswith('elif') else a-1,
                    b=b,promote=None,head=src[a-1].strip()[:64])); got=True
        elif isinstance(n,(ast.For,ast.While,ast.AsyncFor)):
            if dead(n.body):
                found.append(dict(kind='loop',a=n.lineno,b=n.end_lineno,promote=None,
                    head=src[n.lineno-1].strip()[:64])); got=True
        elif isinstance(n,ast.Try):
            for h in n.handlers:
                if dead(h.body):
                    found.append(dict(kind='handler',a=h.lineno,b=h.end_lineno,
                        promote=None,head=src[h.lineno-1].strip()[:64]))
        if not got: walk(n)      # maximal: don't descend into a taken block
for f in tree.body:
    if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)):
        if set(range(f.body[0].lineno,(f.end_lineno or f.lineno)+1)) & hits:
            walk(f)              # only inside LIVE functions
for d in found: d['n']=real(d['a'],d['b'])
found=[d for d in found if d['n']>=3]
found.sort(key=lambda d:-d['n'])
json.dump(found, open(f'{S}/blocks.json','w'), indent=0)
print(f"{len(found)} maximal dead blocks, {sum(d['n'] for d in found)} real lines\n")
for d in found[:22]:
    p='  promote '+str(d['promote']) if d['promote'] else ''
    print(f"  {d['a']:5d}-{d['b']:<5d} {d['n']:4d} {d['kind']:8s} {d['head'][:52]}{p}")
