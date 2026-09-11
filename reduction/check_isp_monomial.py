#!/usr/bin/env python
"""Is the symmetry action on the 3-dim ISP (numerator) space MONOMIAL?

For each symmetry, the raw matrix gives the linear map D_i -> sum_j M_ij D_j + c_i
on the 11 inverse propagators (8 props, slots 0-7; 3 ISPs, slots 8,9,10). The
propagator rows map to props(+const) only, so the propagator subspace is M-invariant
and M induces a well-defined 3x3 action on the ISP quotient = the block M[8:11,8:11].
That block governs the LEADING numerator behaviour (the ISP->prop and ->const parts
are the lower-weight descent). If every generator's 3x3 ISP block is MONOMIAL
(one +/-1 per row and per column) IN KIRA'S BASIS, then a numerator monomial maps to
+/-(relabelled monomial): canonicalization with numerators is the same relabel+sign
as denominators, with NO basis change needed. Otherwise we must search for a common
monomializing basis (or there's a genuine >=2-dim non-monomial block).
"""
import os, sys
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from symmetry_engine import raw_matrix_to_Mc            # M[i]={j:coeff}, c_nonzero
TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
NIDX = 11; ISP = (8, 9, 10)

# parse raw 'symmetries' into matrix blocks
blocks, cur = [], []
for ln in open(os.path.join(TA, "symmetries")):
    s = ln.rstrip("\n")
    if s.strip() == "":
        if cur: blocks.append(cur); cur = []
    else:
        cur.append(s)
if cur: blocks.append(cur)

def isp_block(b):
    M, _ = raw_matrix_to_Mc(b[3:3 + NIDX])
    return tuple(tuple(M.get(i, {}).get(j, 0) for j in ISP) for i in ISP)

def is_monomial(B):
    n = len(B)
    for row in B:                                       # one nonzero (+/-1) per row
        nz = [v for v in row if v != 0]
        if len(nz) != 1 or abs(nz[0]) != 1: return False
    for j in range(n):                                  # one nonzero per column
        if sum(1 for i in range(n) if B[i][j] != 0) != 1: return False
    return True

blocksISP = {}
for b in blocks:
    if len(b) < 3 + NIDX: continue
    B = isp_block(b)
    blocksISP[B] = blocksISP.get(B, 0) + 1

print(f"raw symmetry matrices: {sum(blocksISP.values())}")
print(f"distinct 3x3 ISP-blocks: {len(blocksISP)}\n")
nmono = 0
for B, cnt in sorted(blocksISP.items()):
    mono = is_monomial(B)
    nmono += mono
    tag = "MONOMIAL  " if mono else "NOT-monom "
    print(f"  {tag} (x{cnt:3d})  rows: {B[0]} {B[1]} {B[2]}")
print(f"\n=> distinct ISP-blocks monomial: {nmono} / {len(blocksISP)}")
print("=> ALL generators monomial in Kira's basis?  ",
      "YES — numerator canonicalization is relabel+sign, NO basis change needed"
      if nmono == len(blocksISP) else "NO — a basis change (or residual block) is required")
