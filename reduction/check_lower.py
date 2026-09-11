#!/usr/bin/env python
"""Are all 57 rule terms strictly lower TOTAL-LEX-weight? And are they lower in the
COARSE (w1,w2) sense (real sector/degree progress) or only lateral (same (w1,w2),
lower |abs|)?  The latter is what proliferates at the same orchestrator level."""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from symmetry_route import symmetry_rule, tw_key
K=(0,-1,2,0,2,-1,0,1,-1,0,0)
def w12(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0))
ks=tw_key(K); kw=w12(K)
print(f"source {list(K)}  total-lex-weight = (w1={kw[0]}, w2={kw[1]}, |abs|={ks[2]})\n")
r=symmetry_rule(K)
n_strict_lex=0; n_coarse_lower=0; n_lateral=0
from collections import Counter
lat_w=Counter()
for t in r:
    assert tw_key(t) < ks, f"NOT strictly lower: {t}"   # the rule's invariant
    n_strict_lex+=1
    tw=w12(t)
    if tw < kw:
        n_coarse_lower+=1
    else:                       # same (w1,w2), lower only in |abs| -> lateral
        n_lateral+=1; lat_w[tw]+=1
print(f"all {len(r)} terms strictly lower TOTAL-LEX-weight?  {n_strict_lex==len(r)}   (asserted, passed)")
print(f"\nbreakdown:")
print(f"  coarse-lower  (lower (w1,w2) = real sector/degree progress) : {n_coarse_lower}")
print(f"  LATERAL       (SAME (w1,w2), only lower |abs|)              : {n_lateral}")
print(f"     lateral terms all sit at (w1,w2) = {dict(lat_w)}")
print(f"\n  the orchestrator advances by COARSE weight (level, r, s); the {n_lateral} lateral")
print(f"  terms do NOT lower (r,s) -> they pile up at the SAME level and each re-expands.")
