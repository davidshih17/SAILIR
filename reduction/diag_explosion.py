import pickle, sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
TRIV=set(int(x) for x in open(BASE+"/results/kira_reduce_161/sectormappings/TA/trivialsector").read().split(",") if x.strip())
def secnum(i):
    n=0
    for k in range(8):
        if i[k]>0: n|=(1<<k)
    return n
d=pickle.load(open(BASE+"/results/ab_symmetry/m2_4prop_dots/design1/reduction.pkl","rb"))
fe={k:v for k,v in d['final_expr'].items() if v!=0}
nm=[k for k in fe if any(True for _ in [1])]  # all (masters already filtered? no)
print(f"final_expr terms: {len(fe)}   total_jobs={d['total_jobs']}   routed={d.get('n_symmetry_routed')}   cache={len(d['cache'])}")
# how many final terms are in trivial (scaleless) sectors?
triv_terms=[k for k in fe if secnum(k) in TRIV]
print(f"final terms in TRIVIAL sectors (=0, should have been dropped): {len(triv_terms)} / {len(fe)}")
# weight distribution of final terms
from collections import Counter
wc=Counter((sum(x for x in k if x>0), sum(-x for x in k if x<0)) for k in fe)
print("final-term (w1,w2) distribution (top 8):")
for w,c in sorted(wc.items(), key=lambda x:-x[1])[:8]:
    print(f"    (r={w[0]},s={w[1]}): {c}")
print("sample trivial-sector final terms:")
for k in triv_terms[:5]: print(f"    I{list(k)}  sector={secnum(k)}")
