#!/usr/bin/env python
"""Can a symmetry RAISE the numerator degree (w2)? Apply every symmetry to
  (a) pure-denominator integrals (w2=0, no numerators at all), and
  (b) degree-1 numerators (w2=1),
across all 4-prop sectors, and record the MAX w2 seen in any image term.
If max stays <= source w2, symmetry never CREATES numerator degree -- it only moves it."""
import sys, os, itertools
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
ND=8
def w2(i): return sum(-x for x in i if x<0)     # numerator degree
def numslots(i): return frozenset(j for j in range(11) if i[j]<0)

worst_from0=0; worst_from1=0; ex_raise=None
n_move_examples=0; move_ex=[]
for combo in itertools.combinations(range(ND),4):
    corner=tuple(1 if k in combo else 0 for k in range(11))
    isp=[j for j in range(ND) if j not in combo]+[8,9,10]
    autos=[t for t in _transforms(corner)]
    # (a) pure denominator (w2=0)
    for t in autos:
        img=image_unsigned(corner,*t)
        if img is None: continue
        for k in img:
            if w2(k)>worst_from0: worst_from0=w2(k)
            if w2(k)>0 and ex_raise is None: ex_raise=("w2=0 source",corner,k)
    # (b) degree-1 numerators (w2=1)
    for j in isp:
        src=tuple(-1 if x==j else corner[x] for x in range(11))
        for t in autos:
            img=image_unsigned(src,*t)
            if img is None: continue
            for k in img:
                if w2(k)>worst_from1: worst_from1=w2(k)
                if w2(k)>1 and ex_raise is None: ex_raise=("w2=1 source",src,k)
                # record a slot-move example (same w2, different slot)
                if w2(k)==1 and numslots(k)!=numslots(src) and len(move_ex)<3 and w2(src)==1:
                    move_ex.append((src,k))

print(f"MAX w2 in any image of a PURE-DENOMINATOR (w2=0) source: {worst_from0}")
print(f"MAX w2 in any image of a DEGREE-1 (w2=1) source:        {worst_from1}")
print()
if worst_from0==0 and worst_from1<=1:
    print("=> symmetry NEVER raises numerator degree. It cannot create a numerator on a")
    print("   pure-denominator integral, and cannot turn degree-1 into degree-2.")
    print("   A numerator can only MOVE to a new slot (same total degree).")
else:
    print("=> symmetry CAN raise numerator degree! example:", ex_raise)
print("\nslot-MOVE examples (w2=1 -> w2=1, numerator relocated to a slot that was empty):")
for s,k in move_ex:
    print(f"   {s}  (num@{sorted(j for j in range(11) if s[j]<0)})  ->  {k}  (num@{sorted(j for j in range(11) if k[j]<0)})")
