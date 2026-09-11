import os, sys
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir.symmetries import parse_symmetries
from symmetry_engine import derive_transform, is_placeholder
TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA"); NIDX=11; ISP=(8,9,10)
recs = (parse_symmetries(os.path.join(TA,"sectorSymmetries"),NIDX,2)
        + parse_symmetries(os.path.join(TA,"sectorRelations"),NIDX,2))

# full-family symmetry = ing permutes ALL 8 propagators (no -1 among slots 0..7)
def is_full_family(r): return all(r.ing[g] != -1 for g in range(8))
def is_monomial(B):
    n=len(B)
    if any(sum(1 for v in row if v!=0)!=1 or any(abs(v)>1 for v in row) for row in B): return False
    return all(sum(1 for i in range(n) if B[i][j]!=0)==1 for j in range(n))

print("source sectors present in records:", sorted({r.source_sector for r in recs}))
ff = [r for r in recs if is_full_family(r)]
print(f"\nfull-family records (ing permutes all 8 props): {len(ff)}")
blocks = {}
for r in ff:
    ls = list(r.loop_substs)
    if is_placeholder(ls):
        print(f"  src{r.source_sector}->tgt{r.target_sector}: placeholder (no loop_subst)"); continue
    M,c = derive_transform(ls)
    B = tuple(tuple(int(M[i].get(j,0)) for j in ISP) for i in ISP)
    blocks[B] = blocks.get(B,0)+1
    print(f"  src{r.source_sector}->tgt{r.target_sector}  ISPblock {B[0]}{B[1]}{B[2]}  perm(props)={[r.ing[g] for g in range(8)]}")
print(f"\ndistinct full-family ISP-blocks: {len(blocks)}")
for B,c in blocks.items():
    print(f"  {'MONOMIAL' if is_monomial(B) else 'NOT-monom'} (x{c})  {B}")
