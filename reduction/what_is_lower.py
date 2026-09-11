#!/usr/bin/env python
"""Show explicitly what P_S*I and (lower) are for num@1 in {0,3,4,5}: reduce it via the
symmetry relations and print each resulting term with its sector (prop-set) and weight,
classified same-sector vs lower-sector."""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def sec(i): return frozenset(k for k in range(ND) if i[k]>0)
def pc(i): return sum(1 for k in range(ND) if i[k]>0)
def w12(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0))
combo=(0,3,4,5); corner=tuple(1 if k in combo else 0 for k in range(11)); S=sec(corner); csec=tuple(1 if k in combo else 0 for k in range(ND))
u=tuple(-1 if k==1 else corner[k] for k in range(11))
print(f"num@1 = {u}, sector {sorted(S)} (prop-count {pc(u)}), weight {w12(u)}\n")
autos=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and tuple(1 if x>0 else 0 for x in next(iter(im)))==csec)(image_unsigned(corner,*t))]
# num@1 is odd under some symmetry: find one giving  sigma(u) = -u + lower
for t in autos:
    img=image_unsigned(u,*t)
    if img is None: continue
    if img.get(u,0)%P==P-1 and len(img)>1:   # -u + lower
        print(f"symmetry gives:  num@1 = sigma(num@1) = -num@1 + (lower)  =>  2*num@1 = (lower)")
        print(f"so num@1 = 1/2 * [ the terms below ]:\n")
        for k,c in sorted(img.items()):
            if k==u: continue
            tag = "SAME sector" if sec(k)==S else (f"SUB-sector {sorted(sec(k))} ({pc(k)}p)" if sec(k)<S else "other")
            print(f"   coeff={c%P:>4}  {k}  weight={w12(k)}  ->  {tag}")
        break
