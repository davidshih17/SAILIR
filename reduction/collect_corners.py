#!/usr/bin/env python
"""Collect the distinct uncovered corners (not in REDU/CACHE/MASTERS) that appear in the
symmetry descents of the low symmetric list_TA integrals, so we can reduce them."""
import sys, os, pickle, re
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def secbits(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
def sector(i): return tuple(k for k in range(ND) if i[k]>0)
def w12(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0))
def corner_of(sec): return tuple(1 if k in sec else 0 for k in range(11))
def inv(a): return pow(a,P-2,P)
_ac={}
def sym_autos(corner):
    if corner in _ac: return _ac[corner]
    cs=secbits(corner); a=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and secbits(next(iter(im)))==cs)(image_unsigned(corner,*t))]
    _ac[corner]=a; return a
red=pickle.load(open("results/meta_reduce/list_TA_reductions.pkl","rb")); REDU=red['reductions']
cache=pickle.load(open("replay/reduction_cache.pkl","rb")); CACHE=cache.get('cache',cache)
rhs=set()
for v in REDU.values(): rhs|=set(v)
MASTERS=rhs-set(REDU)
def descent_expr(I0):
    E={}; queue=[(I0,1)]; seen=0
    while queue and seen<5000:
        J,cf=queue.pop(0); seen+=1
        autos=sym_autos(corner_of(sector(J))); avg={}; napp=0; has_id=False
        for t in autos:
            img=image_unsigned(J,*t)
            if img is None: continue
            napp+=1
            if img=={J:1}: has_id=True
            for k,c in img.items(): avg[k]=avg.get(k,0)+c
        if not has_id: avg[J]=avg.get(J,0)+1; napp+=1
        n=inv(napp); avg={k:(c*n)%P for k,c in avg.items() if c%P}; wj=w12(J)
        for k,c in avg.items():
            if w12(k)==wj: E[k]=(E.get(k,0)+(c*cf))%P
            else: queue.append((k,(c*cf)%P))
    return {k:v for k,v in E.items() if v%P}
ints=[tuple(int(x) for x in re.match(r'TA\[([^\]]+)\]',ln.strip()).group(1).split(','))
      for ln in open("from_federica/list_TA_ispclean_by_weight") if ln.startswith("TA[")]
low=sorted([i for i in ints if i in REDU and len(sym_autos(corner_of(sector(i))))>1 and w12(i)[1]>=1],
           key=lambda i:(w12(i)[0]+w12(i)[1]))
uncov=set()
for I in low[:12]:
    for k in descent_expr(I):
        if k not in REDU and k not in CACHE and k not in MASTERS:
            uncov.add(k)
print(f"distinct uncovered corners across 12 descents: {len(uncov)}")
for c in sorted(uncov): print(f"  {list(c)}  sector {list(sector(c))}  w={w12(c)}")
with open("reduction/uncovered_corners.txt","w") as f:
    for c in sorted(uncov): f.write("TA["+",".join(map(str,c))+"]\n")
print("\nwrote reduction/uncovered_corners.txt")
