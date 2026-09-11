#!/usr/bin/env python
"""Is a sectorSymmetry a value-specific relation or a general (symbolic) index map?
   And does it ever map an ISP/numerator position, or always drop it (ing=-1)?
   Check the ing arrays directly for both families."""
import sys, os
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
from sailir.symmetries import parse_symmetries

CASES = [
    ("pentagonbox TA", "results/kira_reduce_161/sectormappings/TA", 11, 2, (8,9,10)),
    ("gravity GR",     "topology_input/gravity3L/kira_validate/sectormappings/GR", 15, 3, (10,11,12,13,14)),
]

for name, sub, n_idx, n_loops, isp in CASES:
    d = os.path.join(ROOT, sub)
    syms = parse_symmetries(os.path.join(d, "sectorSymmetries"), n_idx, n_loops)
    rels = parse_symmetries(os.path.join(d, "sectorRelations"), n_idx, n_loops)
    print(f"=== {name}: {len(syms)} sectorSymmetries + {len(rels)} sectorRelations ===")
    for label, recs in (("sectorSymmetries", syms), ("sectorRelations", rels)):
        isp_mapped = 0      # records where some ISP position has ing != -1 (would mean ISP is permuted)
        denom_is_perm = 0   # records where the non-(-1) targets are all distinct (a permutation)
        for r in recs:
            if any(r.ing[i] != -1 for i in isp):
                isp_mapped += 1
            tgts = [v for v in r.ing if v != -1]
            if len(tgts) == len(set(tgts)):
                denom_is_perm += 1
        print(f"  [{label}] records mapping an ISP position (ing!=-1 at ISP): {isp_mapped} / {len(recs)}")
        print(f"  [{label}] records whose non(-1) targets form a permutation : {denom_is_perm} / {len(recs)}")
    # show that ing is value-agnostic: it references POSITIONS, not values
    ex = syms[0]
    print(f"  example ing (positions, not values): {ex.ing}")
    print(f"    -> means  I[a0..a{n_idx-1}]  =  I[ b ]  with b[ing[g]]=a_g,  for ANY a_g; ISP slots must be 0")
    print()
