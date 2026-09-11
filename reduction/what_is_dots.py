#!/usr/bin/env python
"""What does the loop relabeling behind the sector-55 placeholder (swaps D1<->D3) actually
do to EVERY propagator? The placeholder's clean swap is only the leading part."""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from symmetry_engine import derive_transform, N
# the reflection that swaps D1=k1^2 <-> D3=(k1+p1+p2)^2  is  k1 -> -(k1+p1+p2)
Mf, cf = derive_transform([('k1','-k1-p1-p2'), ('k2','k2')])
names=['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11']
print("loop relabeling k1 -> -(k1+p1+p2), k2 -> k2  (the reflection behind the placeholder):\n")
for i in range(N):
    terms=" + ".join(f"{v}*{names[j]}" for j,v in sorted(Mf[i].items()))
    cc=f"  (+ const {cf[i]})" if cf[i]!=0 else ""
    kind = "clean permutation" if len(Mf[i])==1 and cf[i]==0 else ("-> COMBINATION" if len(Mf[i])>1 else "-> single+const")
    print(f"  {names[i]:4s} -> {terms}{cc}    [{kind}]")
print("\nD2 is a PRESENT propagator of sector 55 (=D1,D2,D3,D5,D6). Does it map cleanly?")
print(f"  D2 maps to {len(Mf[1])} propagator(s): {'CLEAN' if len(Mf[1])==1 else 'a COMBINATION -> cannot sit in a denominator'}")
