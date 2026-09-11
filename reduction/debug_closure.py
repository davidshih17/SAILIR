#!/usr/bin/env python
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned, P
from sailir.symmetries import sector_of
from symmetry_route import tw_key
T=(1,-1,1,0,1,1,0,0,0,0,0)
SPUR=(0,0,0,0,1,1,0,1,0,0,0)
# replicate the closure BFS, find every (K, transform) whose image contains SPUR
seen={T}; frontier=[T]
hits=[]
while frontier:
    K=frontier.pop()
    for (M,c) in _transforms(K):
        img=image_unsigned(K,M,c)
        if img is None: continue
        if SPUR in img:
            hits.append((K,dict(M),dict(c),img))
        for J in img:
            if J not in seen: seen.add(J); frontier.append(J)
print(f"closure size {len(seen)}; SPUR={SPUR} sector={sector_of(SPUR)}")
print(f"\n{len(hits)} (K,transform) pairs image onto SPUR:")
for K,M,c,img in hits[:6]:
    print(f"\n  K={K} sector={sector_of(K)}")
    print(f"    image={img}")
    print(f"    M: "+", ".join(f"D{i}->{M[i]}" for i in sorted(M)))
    nz_c={i:c[i] for i in c if c[i]}
    print(f"    c(nonzero)={nz_c}")
# Also: is SPUR a survivor? does anything reduce it?
print(f"\n=== SPUR's own images (is it symmetric / reducible?) ===")
for (M,c) in _transforms(SPUR):
    img=image_unsigned(SPUR,M,c)
    if img and img!={SPUR:1}: print(f"    SPUR -> {img}")
