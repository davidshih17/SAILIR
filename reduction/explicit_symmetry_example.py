#!/usr/bin/env python
"""Print ONE fully explicit symmetry relation: a chosen integral, its image under
one symmetry as a linear combination with exact coefficients (symbolic in the
Mandelstams, and numeric mod 1009 at a fixed kinematic point)."""
import sys, os
import sympy as sp
sys.path.insert(0, "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2")
sys.path.insert(0, "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/reduction")
from sailir.symmetries import parse_symmetries
from symmetry_engine import derive_transform, N, s12, s23, s34, s45, s51

def transform_full(a, M, c):
    """Image of integral I=prod 1/D_i^{a_i} under D_i -> sum_j M_ij D_j + c_i.
    Returns {index_tuple: sympy_coeff}."""
    base = [0] * N
    sign = sp.Integer(1)
    num_factors = []
    for i, ai in enumerate(a):
        if ai > 0:
            row = M[i]
            assert len(row) == 1, f"denominator pos {i} -> combination {row}"
            j, co = next(iter(row.items()))
            base[j] += ai
            sign *= co**(-ai)                       # 1/(co D_j)^ai
        elif ai < 0:
            num_factors.append((-ai, M[i], c[i]))
    result = {tuple(base): sp.expand(sign)}
    for power, row, const in num_factors:
        for _ in range(power):
            new = {}
            for integ, co in result.items():
                for j, mij in row.items():          # pick D_j (numerator: index j -= 1)
                    ni = list(integ); ni[j] -= 1
                    ni = tuple(ni)
                    new[ni] = sp.expand(new.get(ni, 0) + co * mij)
                if const != 0:                      # pick the constant
                    new[integ] = sp.expand(new.get(integ, 0) + co * const)
            result = {k: v for k, v in new.items() if sp.expand(v) != 0}
    return result

def fmt(a):
    return "TA[" + ",".join(str(x) for x in a) + "]"

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
syms = parse_symmetries(os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA/sectorSymmetries"), N, 2)
rec = next(r for r in syms if r.source_sector == 53)        # a sector-53 symmetry
M, c = derive_transform(list(rec.loop_substs))

I = (1, 0, 1, 0, 2, 1, 0, 0, 0, -1, 0)                       # denoms {0,2,4,5}, DOT at slot 4, numerator -1 at slot 9
print(f"symmetry: loop_subst = {rec.loop_substs}")
print(f"integral I = {fmt(I)}   (sector 53, a dot in slot 4 + one numerator)\n")

img = transform_full(I, M, c)
print("Raw symmetry identity   I = (its relabeled image):")
print(f"  {fmt(I)}  =")
for J, co in sorted(img.items()):
    print(f"      ({sp.simplify(co)}) * {fmt(J)}")

# Solve for I (it appears on the RHS with some coefficient)
selfco = img.get(I, sp.Integer(0))
print(f"\nI appears on the RHS with coefficient {selfco}; move it over and solve for I:")
denom = sp.simplify(1 - selfco)
print(f"  {fmt(I)}  =")
for J, co in sorted(img.items()):
    if J == I:
        continue
    print(f"      ({sp.simplify(co/denom)}) * {fmt(J)}")

# numeric version mod 1009 at a fixed kinematic point
P = 1009
subsK = {s12: 2, s23: 3, s34: 5, s45: 7, s51: 11}
print(f"\nNumeric (mod {P}, at s12,s23,s34,s45,s51 = 2,3,5,7,11) -- what SAILIR stores:")
print(f"  {fmt(I)}  =")
for J, co in sorted(img.items()):
    if J == I:
        continue
    val = sp.simplify((co / denom).subs(subsK))      # substitute kinematics first
    num, den = sp.fraction(sp.together(val))
    coeff_modp = (int(num) % P) * pow(int(den) % P, P - 2, P) % P
    print(f"      {coeff_modp:>4d} * {fmt(J)}")
