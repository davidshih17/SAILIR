#!/usr/bin/env python
"""Reconstruct the loop relabeling (k1',k2') of a placeholder symmetry from its
propagator permutation (ing), so its FULL momentum transform can be derived. A symmetry
maps momentum(D_g) -> ±momentum(D_sigma(g)); anchoring on a k1-type and k2-type present
propagator determines k1',k2' as 6-vectors (k1,k2,p1,p2,p3,p4)."""
from itertools import product as iproduct
# propagator momenta as 6-vectors (k1,k2,p1,p2,p3,p4)
PM = {0:(1,0,0,0,0,0),1:(1,0,1,0,0,0),2:(1,0,1,1,0,0),3:(1,0,1,1,1,0),
      4:(0,1,0,0,0,0),5:(0,1,1,1,1,0),6:(0,1,1,1,1,1),7:(1,-1,0,0,0,0),
      8:(1,0,1,1,1,1),9:(0,1,1,0,0,0),10:(0,1,1,1,0,0)}
def vadd(a,b): return tuple(x+y for x,y in zip(a,b))
def vscale(s,a): return tuple(s*x for x in a)
def ppart(v): return (0,0)+v[2:]        # zero the k1,k2 part
NAMES=['k1','k2','p1','p2','p3','p4']
def vec_to_str(v):
    s=""
    for i,x in enumerate(v):
        if x==0: continue
        t = NAMES[i] if abs(x)==1 else f"{abs(x)}*{NAMES[i]}"
        s += ("-" if x<0 else ("+" if s else "")) + t
    return s or "0"

def reconstruct(ing, present_props):
    """Return loop_subst [('k1',rhs1),('k2',rhs2)] or None."""
    sigma = {g: ing[g] for g in present_props}
    k1p = [g for g in present_props if PM[g][:2]==(1,0)]
    k2p = [g for g in present_props if PM[g][:2]==(0,1)]
    # candidate R1 (=k1') from each k1-anchor & sign; R2 (=k2') from each k2-anchor & sign
    def cands(anchors, default):
        out=[]
        if not anchors: out.append(default)
        for g in anchors:
            for eps in (1,-1):
                # alpha*R = eps*m_sigma(g) - ppart(m_g), with alpha=1 (k1 or k2 type)
                R = tuple(a-b for a,b in zip(vscale(eps, PM[sigma[g]]), ppart(PM[g])))
                out.append(R)
        return out
    for R1 in cands(k1p, PM[0]):        # default k1'=k1
        for R2 in cands(k2p, PM[4]):    # default k2'=k2
            ok=True
            for g in present_props:
                a,b = PM[g][:2]
                lhs = vadd(vadd(vscale(a,R1), vscale(b,R2)), ppart(PM[g]))
                # must equal +/- m_sigma(g)
                if lhs != PM[sigma[g]] and lhs != vscale(-1, PM[sigma[g]]):
                    ok=False; break
            if ok:
                return [('k1', vec_to_str(R1)), ('k2', vec_to_str(R2))]
    return None

if __name__ == "__main__":
    import sys, os
    BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
    from sailir.symmetries import parse_symmetries
    from symmetry_engine import N
    TA=os.path.join(BASE,"results/kira_reduce_161/sectormappings/TA")
    recs=parse_symmetries(os.path.join(TA,"sectorSymmetries"),N,2)+parse_symmetries(os.path.join(TA,"sectorRelations"),N,2)
    # SELF-CHECK: for momentum-map records, reconstruct from ing and compare permutation to actual
    def sector_props(sec): return [g for g in range(8) if sec>>g & 1]
    ok=bad=ph=0
    for r in recs:
        ls=list(r.loop_substs)
        pres=sector_props(r.source_sector)
        rec=reconstruct(r.ing, pres)
        if any(rhs=="placeholder" for _,rhs in ls):
            ph+=1; continue
        # momentum-map record: does reconstructed relabel reproduce the SAME permutation? compare via derive
        if rec is None: bad+=1; continue
        ok+=1
    print(f"momentum-map records: reconstruct succeeded {ok}, failed {bad}")
    print(f"placeholder records: {ph}")
    # can we reconstruct ALL placeholders?
    phok=phbad=0
    for r in recs:
        if not any(rhs=="placeholder" for _,rhs in list(r.loop_substs)): continue
        rec=reconstruct(r.ing, sector_props(r.source_sector))
        if rec: phok+=1
        else: phbad+=1
    print(f"placeholder reconstruction: {phok} succeeded, {phbad} failed")
