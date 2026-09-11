#!/usr/bin/env python
import pickle, sys, os, glob
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE)
# trivial sectors for annotation
TRIV=set(int(x) for x in open(BASE+"/results/kira_reduce_161/sectormappings/TA/trivialsector").read().split(",") if x.strip())
def secnum(i): 
    n=0
    for k in range(8):
        if i[k]>0: n|=(1<<k)
    return n
d=pickle.load(open(BASE+"/results/ab_symmetry/gate_small/baseline/reduction.pkl","rb"))
print("=== baseline final_expr ===", {k:v for k,v in d['final_expr'].items()})
print(f"total_jobs={d['total_jobs']}, cache size={len(d['cache'])}\n")
print("=== baseline cache (each worker's one-step reduction) ===")
for I,red in d['cache'].items():
    s=secnum(I); triv=" [TRIVIAL sector=%d]"%s if s in TRIV else ""
    print(f"  I{list(I)}{triv}  sector={s}")
    if not red:
        print(f"      -> {{}}  (REDUCES TO ZERO)")
    else:
        for k,c in red.items():
            sk=secnum(k); tk=" [TRIV]" if sk in TRIV else ""
            print(f"      -> {c} * I{list(k)}{tk}")
