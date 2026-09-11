#!/usr/bin/env python
"""Check a design1 reduction.pkl's masters against the meta_reduce ground truth."""
import sys, pickle
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
d=pickle.load(open(sys.argv[1],"rb"))
meta=pickle.load(open(BASE+"/results/meta_reduce/list_TA_reductions.pkl","rb"))
start=tuple(d['start_integral'])
md={i:c for i,c in d['final_expr'].items() if c!=0}
gt=meta['reductions'].get(start)
print(f"target I{list(start)}")
print(f"design1 masters ({len(md)}):")
for i,c in sorted(md.items()): print(f"    I{list(i)} = {c}")
if gt is None:
    print("  NOT in meta_reduce GT")
else:
    gt={tuple(k):v for k,v in gt.items()}
    print(f"GT masters ({len(gt)}):")
    for i,c in sorted(gt.items()): print(f"    I{list(i)} = {c}")
    print(f"\n  MATCH design1==GT : {md==gt}")
