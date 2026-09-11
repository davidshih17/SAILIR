#!/usr/bin/env python
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned, P
from sailir.symmetries import sector_of
T=(1,-1,1,0,1,1,0,0,0,0,0)
SPUR=(0,0,0,0,1,1,0,1,0,0,0)
print(f"target {T}  sector={sector_of(T)}  (props D1,D3,D5,D6 = positions 0,2,4,5)\n")
for n,(M,c) in enumerate(_transforms(T)):
    img=image_unsigned(T,M,c)
    if img is None: continue
    mark=" <==SPUR" if SPUR in img else ""
    print(f"transform#{n}: image = {img}{mark}")
    if SPUR in img:
        print(f"   M (D_i -> combos):")
        for i in sorted(M): print(f"     D{i} -> {M[i]}   c={c.get(i,0)}")
