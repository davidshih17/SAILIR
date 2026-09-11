import os, sys
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir.symmetries import parse_symmetries
from symmetry_engine import derive_transform, is_placeholder
TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA"); NIDX = 11; ISP = (8, 9, 10)

recs = (parse_symmetries(os.path.join(TA, "sectorSymmetries"), NIDX, 2)
        + parse_symmetries(os.path.join(TA, "sectorRelations"), NIDX, 2))
def is_monomial(B):
    n = len(B)
    for row in B:
        nz = [v for v in row if v != 0]
        if len(nz) != 1 or abs(nz[0]) != 1: return False
    for j in range(n):
        if sum(1 for i in range(n) if B[i][j] != 0) != 1: return False
    return True

blocks = {}; n_used = n_ph = 0; err = 0
for r in recs:
    ls = list(r.loop_substs)
    if is_placeholder(ls): n_ph += 1; continue
    try:
        M, c = derive_transform(ls)          # globally-consistent momentum-algebra transform
    except Exception as e:
        err += 1; continue
    B = tuple(tuple(int(M[i].get(j, 0)) for j in ISP) for i in ISP)
    blocks[B] = blocks.get(B, 0) + 1
    n_used += 1
print(f"records: {len(recs)}  used(real loop_subst): {n_used}  placeholder(skipped): {n_ph}  errors: {err}")
print(f"distinct GLOBAL 3x3 ISP-blocks: {len(blocks)}\n")
nmono = 0
for B, cnt in sorted(blocks.items(), key=lambda kv: -kv[1]):
    mono = is_monomial(B); nmono += mono
    print(f"  {'MONOMIAL ' if mono else 'NOT-monom'} (x{cnt:3d})  {B[0]} {B[1]} {B[2]}")
print(f"\n=> monomial: {nmono}/{len(blocks)}  -> ALL monomial? "
      f"{'YES (numerator canon = relabel+sign, no basis change)' if nmono==len(blocks) else 'NO'}")
