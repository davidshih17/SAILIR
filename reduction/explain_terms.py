#!/usr/bin/env python
"""Show EXACTLY what the many-term symmetry images/rules are, and why they blow up."""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned, P
from sailir.symmetries import sector_of

def show(i): return "I"+str(list(i))

# ---------- 1) the SIMPLEST mechanism: ONE numerator -> a combination ----------
print("="*70)
print("(1) MECHANISM: a numerator maps to a COMBINATION, not a single slot")
print("="*70)
# integral with a single numerator D9 (ISP, position 8, power -1), denominators D1 D3 D5 D6
J=(1,0,1,0,1,1,0,0,-1,0,0)
print(f"take {show(J)}  = D1 D3 D5 D6 * (D9)^1  in the numerator  (sector {sector_of(J)})")
for n,(M,c) in enumerate(_transforms(J)):
    img=image_unsigned(J,M,c)
    if img is None: continue
    if len(img)>=3:                      # find one that expands the numerator
        print(f"\n  under symmetry #{n}, the numerator slot D9 maps to:  M[8] = {M.get(8)}   const c[8]={c.get(8,0)}")
        print(f"  so D9 (upstairs) becomes that linear combination, and the image is {len(img)} terms:")
        for k,co in sorted(img.items()): print(f"      {co:>4} * {show(k)}")
        break

# ---------- 2) DEGREE MULTIPLIES: 3 numerators each -> combination ----------
print("\n"+"="*70)
print("(2) WHY IT EXPLODES: numerator DEGREE multiplies (each factor expands)")
print("="*70)
K=(0,-1,2,0,2,-1,0,1,-1,0,0)   # the source of the 57-term rule
nums=[p for p in range(11) if K[p]<0]
print(f"the 57-term rule's source: {show(K)}  (sector {sector_of(K)})")
print(f"  denominators: D3^2 D5^2 D8   ;   numerators: D2^1 D6^1 D9^1  (degree s = 3)")
# find a transform and show each numerator's image-combination, then the product size
best=None
for (M,c) in _transforms(K):
    img=image_unsigned(K,M,c)
    if img is None: continue
    if best is None or len(img)>len(best[0]): best=(img,M,c)
img,M,c=best
print(f"\n  under one symmetry, the three numerator slots map to combinations:")
for p in nums:
    row=M.get(p,{}); combo=" + ".join(f"{v}*D{j}" for j,v in row.items()) or "0"
    print(f"      D{p+1} (slot {p})  ->  {combo}   (+{c.get(p,0)})   [{len(row)+ (1 if c.get(p,0) else 0)} choices]")
print(f"\n  the numerator is a PRODUCT of these -> the expansion is a multinomial:")
print(f"  this single symmetry image already has {len(img)} distinct integrals. Sample:")
for k,co in list(sorted(img.items()))[:8]:
    print(f"      {co:>4} * {show(k)}")
print(f"      ... ({len(img)} total)")

# ---------- 3) the FINAL RREF rule (accumulated over the orbit) ----------
print("\n"+"="*70)
print("(3) the RULE symmetry_route builds for it (RREF over the whole orbit closure)")
print("="*70)
from symmetry_route import symmetry_rule
r=symmetry_rule(K)
if r is None:
    print("  survivor")
else:
    print(f"  {show(K)}  ->  {len(r)} lower integrals. Sample:")
    for k,co in list(sorted(r.items()))[:8]:
        print(f"      {co:>4} * {show(k)}   (sector {sector_of(k)})")
    print(f"      ... ({len(r)} total)")
