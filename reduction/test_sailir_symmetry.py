#!/usr/bin/env python
import sys, os, time
sys.path.insert(0, "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/reduction")
from sailir_symmetry import SymmetryReducer, _target_key, _weight
from sailir.symmetries import sector_of

t0 = time.time()
R = SymmetryReducer()
print(f"built reducer in {time.time()-t0:.1f}s; sectors with symmetries: {len(R.by_source)}\n")

def fmt(a): return "TA[" + ",".join(map(str, a)) + "]"

PROBES = {
    "(8,4)":       (-1, 2, 1, 0, 1, 2, 1, 1, -3, 0, 0),
    "(7,4)":       (0, 1, 1, 1, 1, 1, 1, 1, -4, 0, 0),
    "long-runner": (1, 1, 1, 0, 0, 1, 3, 1, -2, -1, 0),
}
print("=== probe integrals: symmetry-reducible at the top? ===")
for name, I in PROBES.items():
    s = R.sol(I)
    print(f"  {name:12s} {fmt(I)}  sector={sector_of(I)} weight={_weight(I)[:2]}  "
          f"reducible={s is not None}" + (f"  -> {len(s)} terms" if s else ""))

# verify a known-reducible example: the dotted sector-53 integral from before
print("\n=== verify a sol is a correct relation (sector-53 dotted+numerator) ===")
I = (1, 0, 1, 0, 2, 1, 0, 0, 0, -1, 0)
s = R.sol(I)
if s is None:
    print(f"  {fmt(I)} not reducible (I is not the pivot of its orbit at this order)")
else:
    print(f"  {fmt(I)}  =")
    for J, co in sorted(s.items()):
        print(f"      {co:>5d} * {fmt(J)}   [tkey {'<' if _target_key(J) > _target_key(I) else '>='} I]")

# also try its partner (the other dot position) — exactly one of the pair should reduce
Ip = (1, 0, 1, 0, 1, 2, 0, 0, 0, -1, 0)
print(f"\n  partner {fmt(Ip)} reducible={R.sol(Ip) is not None}")
print(f"  (exactly one of the same-weight pair should be the pivot)")
