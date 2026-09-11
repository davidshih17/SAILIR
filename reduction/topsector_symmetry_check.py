#!/usr/bin/env python
"""Do loop relabelings permute the TOP sector's 8 propagators onto themselves?
Represent each D_i momentum as (k1,k2,p1,p2,p3,p4). Apply candidate relabelings and see
what each of D1..D8 maps to (a propagator? an ISP slot? a non-propagator?)."""
# momentum vectors: (c_k1, c_k2, c_p1, c_p2, c_p3, c_p4)
D = {
 1:(1,0,0,0,0,0),        2:(1,0,1,0,0,0),   3:(1,0,1,1,0,0),   4:(1,0,1,1,1,0),   # k1 side (props)
 5:(0,1,0,0,0,0),        6:(0,1,1,1,1,0),   7:(0,1,1,1,1,1),                        # k2 side (props)
 8:(1,-1,0,0,0,0),                                                                  # rung (prop)
 9:(1,0,1,1,1,1),       10:(0,1,1,0,0,0),  11:(0,1,1,1,0,0),                        # ISP slots 8,9,10
}
PROPS=set(range(1,9)); ISP={9,10,11}
def neg(v): return tuple(-x for x in v)
def matchD(v):
    for i,dv in D.items():
        if v==dv or v==neg(dv): return i
    return None
def add(*vs): return tuple(sum(c) for c in zip(*vs))
def scale(a,v): return tuple(a*x for x in v)
def apply(relabel, v):
    # relabel = (new_k1_vec, new_k2_vec); externals pass through
    nk1,nk2=relabel; ck1,ck2=v[0],v[1]; ext=(0,0)+v[2:]
    return add(scale(ck1,nk1), scale(ck2,nk2), ext)

k1=(1,0,0,0,0,0); k2=(0,1,0,0,0,0)
relabels={
 "k1<->k2 (loop exchange)": (k2, k1),
 "k1 reflection":           ((-1,0,-1,-1,-1,0), k2),          # k1 -> -k1-p1-p2-p3
 "k2 reflection":           (k1, (0,-1,-1,-1,-1,0)),          # k2 -> -k2-p1-p2-p3
}
for name,rl in relabels.items():
    print(f"\n=== {name} ===")
    images={}; ok=True
    for i in sorted(PROPS):
        vp=apply(rl, D[i]); j=matchD(vp)
        where = f"D{j}" + (" [ISP slot!]" if j in ISP else "") if j else "NOT a D (breaks)"
        images[i]=j
        print(f"  D{i} -> {where}")
    imgset={images[i] for i in PROPS}
    preserved = imgset==PROPS
    print(f"  image of top-sector props {{D1..D8}} = {sorted(x for x in imgset if x)}"
          + ("" if all(images[i] for i in PROPS) else " + non-D"))
    print(f"  PRESERVES top sector {{D1..D8}}? {preserved}")
