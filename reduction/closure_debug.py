#!/usr/bin/env python
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
def sym_autos(corner):
    cs=secbits(corner)
    return [t for t in _transforms(corner) if (lambda im: im and len(im)==1 and secbits(next(iter(im)))==cs)(image_unsigned(corner,*t))]
red=pickle.load(open("results/meta_reduce/list_TA_reductions.pkl","rb")); REDU=red['reductions']
ints=[tuple(int(x) for x in re.match(r'TA\[([^\]]+)\]',ln.strip()).group(1).split(','))
      for ln in open("from_federica/list_TA_ispclean_by_weight") if ln.startswith("TA[")]

# problematic one
I=(1,1,1,-3,1,1,1,0,0,-2,0)
autos=sym_autos(corner_of(sector(I)))
print(f"I={I}  sector {list(sector(I))}  #autos={len(autos)}")
imgs=[image_unsigned(I,*t) for t in autos]
print(f"  image_unsigned(I,t): {sum(1 for im in imgs if im is None)} None / {len(imgs)}")
for im in imgs[:3]: print("   ", None if im is None else f"{len(im)} terms, e.g. {list(im.items())[:1]}")

# low-weight symmetric list_TA candidates in REDU
low=sorted([i for i in ints if i in REDU and len(sym_autos(corner_of(sector(i))))>1 and w12(i)[1]>=1],
           key=lambda i:(w12(i)[0]+w12(i)[1]))
print(f"\nlowest-weight symmetric list_TA integrals in REDU:")
for i in low[:6]:
    autos=sym_autos(corner_of(sector(i)))
    nNone=sum(1 for t in autos if image_unsigned(i,*t) is None)
    print(f"  {i}  w={w12(i)}  #autos={len(autos)}  image-None={nNone}/{len(autos)}")
