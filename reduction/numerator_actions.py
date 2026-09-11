#!/usr/bin/env python
"""Enumerate what a symmetry can do to a numerator. A numerator index j means D_j
(a quadratic in loop momenta) sits upstairs. Under a relabeling k->Mk+c, D_j becomes a
new quadratic, re-expanded in the D_1..D_11 basis + constant. Every image term falls
into exactly one category. We apply each symmetry to single-numerator integrals in a
few sectors and TALLY the categories."""
import sys, os, itertools
from collections import Counter
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
ND=8
def props(i): return frozenset(k for k in range(ND) if i[k]>0)
def pc(i): return sum(1 for k in range(ND) if i[k]>0)
def numdeg(i): return sum(-x for x in i if x<0)     # total numerator degree (w2)
def dots(i): return sum(x-1 for k,x in enumerate(i[:ND]) if x>0)  # extra dots

def classify_term(src, k):
    """category of image term k relative to source numerator integral src"""
    ps, pk = props(src), props(k)
    ds, dk = numdeg(src), numdeg(k)
    if pk==ps and dk==ds and dots(k)==dots(src):
        # same denominators, same numerator degree, same dots -> numerator moved (ISP->ISP)
        nsrc=[j for j in range(11) if src[j]<0]; nk=[j for j in range(11) if k[j]<0]
        return "ISP->ISP same-slot" if nk==nsrc else "ISP->ISP other-slot(s)"
    if pk < ps:
        return "denominator cancelled -> LOWER sector"
    if pk==ps and dk<ds:
        return "numerator -> constant -> LOWER degree (w2 down)"
    if pk==ps and dots(k)<dots(src) and dk==ds:
        return "numerator hit a dotted prop -> fewer dots"
    if pk==ps and dk<ds:
        return "LOWER degree"
    return f"other (pc {pc(k)} deg {dk})"

tally=Counter(); examples={}
for combo in [(0,3,4,5),(0,1,2,3),(1,2,4,7)]:
    corner=tuple(1 if k in combo else 0 for k in range(11)); csec=tuple(1 if k in combo else 0 for k in range(ND))
    isp_slots=[j for j in range(ND) if j not in combo]+[8,9,10]
    autos=[t for t in _transforms(corner)]
    for j in isp_slots:
        src=tuple(-1 if k==j else corner[k] for k in range(11))   # corner + numerator on slot j
        for t in autos:
            img=image_unsigned(src,*t)
            if img is None: continue
            for k in img:
                c=classify_term(src,k); tally[c]+=1
                if c not in examples and props(src)==frozenset(combo):
                    examples[c]=(src,k)

print("=== ALL categories a symmetry produces from a numerator (tallied over sectors {0,3,4,5},{0,1,2,3},{1,2,4,7}) ===\n")
for cat,n in tally.most_common():
    print(f"  {n:5d}   {cat}")
print("\n=== one concrete example of each ===")
for cat,(src,k) in examples.items():
    print(f"  [{cat}]")
    print(f"       {src}   -->  ... {k} ...")
