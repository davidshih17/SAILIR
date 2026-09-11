#!/usr/bin/env python
"""(1) sector distribution of the 996 list_TA. (2) symmetry descent
I = P_S.I + P_S.I' + ... + corners, for a top-sector list_TA integral (trivial) and a
symmetric-sector integral (non-trivial)."""
import sys, os, re
from collections import Counter
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def secbits(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
def sector(i): return tuple(k for k in range(ND) if i[k]>0)
def pc(i): return sum(1 for k in range(ND) if i[k]>0)
def w12(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0))
def corner_of(sec): return tuple(1 if k in sec else 0 for k in range(11))
def inv(a): return pow(a,P-2,P)
_ac={}
def sym_autos(corner):
    if corner in _ac: return _ac[corner]
    cs=secbits(corner)
    a=[t for t in _transforms(corner)
       if (lambda im: im and len(im)==1 and secbits(next(iter(im)))==cs)(image_unsigned(corner,*t))]
    _ac[corner]=a; return a

# parse list_TA
ints=[]
for ln in open("from_federica/list_TA_ispclean_by_weight"):
    m=re.match(r'TA\[([^\]]+)\]',ln.strip())
    if m: ints.append(tuple(int(x) for x in m.group(1).split(',')))
print(f"list_TA_ispclean: {len(ints)} integrals")
bypc=Counter(pc(i) for i in ints)
symcnt=Counter()
for i in ints:
    if len(sym_autos(corner_of(sector(i))))>1: symcnt[pc(i)]+=1
print("by prop-count (count, #in symmetric sector):")
for p in sorted(bypc): print(f"  {p}-prop: {bypc[p]:4d}   symmetric-sector: {symcnt[p]}")
print(f"  TOTAL in symmetric sectors: {sum(symcnt.values())} / {len(ints)}\n")

# symmetry descent
def descent(I0, tag):
    print(f"===== symmetry descent of {tag}: {I0}  sector {list(sector(I0))} ({pc(I0)}p) w(r,s)={w12(I0)} =====")
    terms=[]; queue=[(I0,1)]; steps=0
    while queue and steps<40:
        J,coef=queue.pop(0); steps+=1
        autos=sym_autos(corner_of(sector(J)))
        avg={}
        for t in autos:
            img=image_unsigned(J,*t)
            if img is None: continue
            for k,c in img.items(): avg[k]=avg.get(k,0)+c
        n=inv(max(len(autos),1)); avg={k:(c*n)%P for k,c in avg.items() if c%P}
        wj=w12(J)
        PS={k:(c*coef)%P for k,c in avg.items() if w12(k)==wj}
        R={k:(c*coef)%P for k,c in avg.items() if w12(k)!=wj}
        term_kind = "TERMINAL (asym/corner)" if not R else "P_S term"
        terms.append((J,coef,PS,term_kind))
        for k,c in R.items(): queue.append((k,c))
    # print the chain
    for J,coef,PS,kind in terms:
        pss=" + ".join(f"{c}*{k}" for k,c in list(PS.items())[:4])
        more="" if len(PS)<=4 else f" +{len(PS)-4}more"
        print(f"  [{kind}] P_S.({coef}*{J}) = {pss}{more}   ({len(PS)} terms)")
    print(f"  ... {len(terms)} P_S contributions, {steps} descent steps\n")

# top-sector list_TA integral (trivial) and lower one
top=ints[0]
descent(top, "top list_TA")
# a symmetric-sector integral: sector 57={0,3,4,5} with a numerator (non-trivial)
sym_ex=(1,0,0,1,1,1,0,0,-1,0,0)   # corner {0,3,4,5} + numerator on slot 8 (D9)
descent(sym_ex, "sector-57 numerator example")
