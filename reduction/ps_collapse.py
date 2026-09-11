#!/usr/bin/env python
"""Do the 57 RHS terms collapse when symmetrized with P_S = (1/|G|) sum_g M_g?
Two terms J,J' merge iff P_S.J == P_S.J' (as symmetric combinations). Count distinct."""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned, P
from symmetry_route import symmetry_rule, tw_key

K=(0,-1,2,0,2,-1,0,1,-1,0,0)
rule=symmetry_rule(K)
terms=list(rule.keys())
print(f"rule for {list(K)} has {len(terms)} terms\n")

def inv(a): return pow(a,P-2,P)
def normed(vec):
    """Projective normalization: scale so the lex-first nonzero coeff = 1 -> hashable key."""
    vec={k:v%P for k,v in vec.items() if v%P}
    if not vec: return ()
    k0=min(vec); s=inv(vec[k0])
    return tuple(sorted((k,(v*s)%P) for k,v in vec.items()))

# (A) clean-permutation orbit rep (dedup notion: single-integral relabelings only)
def clean_rep(J):
    seen={J}; fr=[J]
    while fr:
        x=fr.pop()
        for (M,c) in _transforms(x):
            img=image_unsigned(x,M,c)
            if img and len(img)==1:
                y=next(iter(img))
                if y not in seen: seen.add(y); fr.append(y)
    return min(seen)
clean_orbits={clean_rep(J) for J in terms}

# (B) full P_S symmetrization: sym(J) = sum over ALL applicable transforms of image(J)
def sym(J):
    acc={}
    for (M,c) in _transforms(J):
        img=image_unsigned(J,M,c)
        if img is None: continue
        for k,v in img.items(): acc[k]=(acc.get(k,0)+v)%P
    return normed(acc)
ps_classes={}
zero=0
for J in terms:
    s=sym(J)
    if not s: zero+=1; continue
    ps_classes.setdefault(s, []).append(J)

print(f"(A) distinct CLEAN-permutation orbits among the 57 terms : {len(clean_orbits)}")
print(f"(B) distinct P_S images (full symmetrization)            : {len(ps_classes)}   (+{zero} that P_S sends to 0)")
print(f"\nso reducing P_S.I instead of I: the {len(terms)}-term RHS collapses to "
      f"{len(ps_classes)} symmetric terms ({zero} vanish).")
# show a P_S class that merged several terms
for s,js in sorted(ps_classes.items(), key=lambda kv:-len(kv[1]))[:3]:
    if len(js)>1:
        print(f"   merged {len(js)} terms into one P_S image, e.g. {[list(j) for j in js[:3]]}")
