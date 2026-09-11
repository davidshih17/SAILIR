#!/usr/bin/env python
"""Clean symmetry descent I = P_S.I + P_S.I' + ... + corners for a REAL list_TA integral
in a symmetric sector."""
import sys, os, re
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
def iscorner(i): return all(i[k] in (0,1) for k in range(ND)) and all(i[k]==0 for k in range(8,11))
def corner_of(sec): return tuple(1 if k in sec else 0 for k in range(11))
def inv(a): return pow(a,P-2,P)
_ac={}
def sym_autos(corner):
    if corner in _ac: return _ac[corner]
    cs=secbits(corner)
    a=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and secbits(next(iter(im)))==cs)(image_unsigned(corner,*t))]
    _ac[corner]=a; return a
def coef(c): 
    c%=P
    return str(c) if c<=P//2 else f"-{P-c}"

def descent(I0):
    print(f"===== descent of {I0}   sector {list(sector(I0))} ({pc(I0)}p)  (r,s)={w12(I0)} =====")
    queue=[(I0,1)]; chain=[]; steps=0
    while queue and steps<60:
        J,cf=queue.pop(0); steps+=1
        autos=sym_autos(corner_of(sector(J)))
        if len(autos)<=1:   # asymmetric OR only identity: P_S = id, terminal
            chain.append(("TERMINAL(asym)", J, cf, {J:cf})); continue
        avg={}
        for t in autos:
            img=image_unsigned(J,*t)
            if img is None: continue
            for k,c in img.items(): avg[k]=avg.get(k,0)+c
        n=inv(len(autos)); avg={k:(c*n)%P for k,c in avg.items() if c%P}
        wj=w12(J)
        PS={k:(c*cf)%P for k,c in avg.items() if w12(k)==wj}
        R={k:(c*cf)%P for k,c in avg.items() if w12(k)!=wj}
        if not R:  # J fully symmetric at its level (e.g. corner) -> terminal
            chain.append(("TERMINAL(corner/sym)", J, cf, PS)); continue
        chain.append(("P_S.J", J, cf, PS))
        for k,c in R.items(): queue.append((k,c))
    for kind,J,cf,PS in chain:
        body = "0  (no symmetric survivor)" if not PS else " + ".join(f"{coef(c)}*{k}" for k,c in list(PS.items())[:5])
        print(f"  [{kind:20s}] P_S({coef(cf)}*{J}) = {body}")
    # collect the terminal corner combination
    corners={}
    for kind,J,cf,PS in chain:
        if kind.startswith("TERMINAL"):
            for k,c in PS.items(): corners[k]=(corners.get(k,0)+c)%P
    print(f"\n  => terminates at {len(corners)} corner/terminal integrals; {steps} descent steps")
    print(f"  I = (P_S survivors, mostly 0 here) + sum of these {len(corners)} corners.\n")

# real list_TA integral in a symmetric sector, with numerators
ints=[]
for ln in open("from_federica/list_TA_ispclean_by_weight"):
    m=re.match(r'TA\[([^\]]+)\]',ln.strip())
    if m: ints.append(tuple(int(x) for x in m.group(1).split(',')))
cand=[i for i in ints if len(sym_autos(corner_of(sector(i))))>1 and w12(i)[1]>=1 and pc(i)==5]
print(f"picking a real 5-prop list_TA integral in a symmetric sector (of {len(cand)} such)\n")
descent(cand[len(cand)//2])
descent(cand[-1])
