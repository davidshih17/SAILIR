import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
ND=8
def sector(i): return tuple(k for k in range(ND) if i[k]>0)
def corner_of(s): return tuple(1 if k in s else 0 for k in range(11))
def secbits(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
# compare a LOW integral (numerators on mapped slots -> applies) vs HIGH (numerators on unmapped slots -> None)
for I in [(1,1,1,-3,1,1,1,0,0,-2,0), (1,-1,1,0,1,1,0,0,0,0,0)]:
    corner=corner_of(sector(I)); cs=secbits(corner)
    autos=[(M,c) for (M,c) in _transforms(corner) if (lambda im: im and len(im)==1 and secbits(next(iter(im)))==cs)(image_unsigned(corner,M,c))]
    numslots=[j for j in range(11) if I[j]<0]
    print(f"I={I}  numerators on slots {numslots}, sector {sector(I)}, {len(autos)} autos:")
    for idx,(M,c) in enumerate(autos):
        imI=image_unsigned(I,M,c)
        mapped=sorted(M.keys()); unmapped=[j for j in range(11) if j not in M]
        num_unmapped=[j for j in numslots if (j not in M or not M[j]) and not c.get(j,0)]
        print(f"   auto{idx}: I-image={'None' if imI is None else str(len(imI))+' terms'}"
              f" | M maps slots {mapped} | numerator slots M can't handle: {num_unmapped}")
    print()
