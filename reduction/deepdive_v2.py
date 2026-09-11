#!/usr/bin/env python
"""DEEP DIVE v2 — rule out bugs / confirm mechanism using the SymmetryReducer:
 (A) are symmetry reductions strictly weight-LOWERING in the total order? (orientation)
 (B) are the ON-only integrals symmetry-PIVOTS whose products land back in the
     shared tree (= 'symmetry detours' / missing orbit-dedup), vs spurious junk?
No theory — calls R.sol() directly to reproduce what the workers did."""
import glob, os, sys
from collections import Counter
sys.path.insert(0, "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/reduction")
from sailir_symmetry import SymmetryReducer, _weight, _target_key

BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
ON, OFF = f"{BASE}/results/probetop84_on", f"{BASE}/results/probetop84_off"

def parse_int(path):
    b = os.path.basename(path)
    for e in (".pkl", ".out"):
        if b.endswith(e): b = b[:-len(e)]
    if not b.startswith("async_"): return None
    t = b.split("_")
    try:
        it = tuple(int(x) for x in t[2:13])
    except ValueError:
        return None
    return it if len(it) == 11 else None

def reduced_set(d):
    return {it for f in glob.glob(f"{d}/work/results/*.pkl") if (it := parse_int(f))}

def sym_solved_set(d):
    s = set()
    for f in glob.glob(f"{d}/work/logs/*.out"):
        it = parse_int(f)
        if it and it not in s:
            try:
                with open(f, errors="ignore") as fh:
                    if any("drained after symmetry" in ln for ln in fh): s.add(it)
            except OSError: pass
    return s

print("loading reducer + sets...", flush=True)
R = SymmetryReducer()
on, off = reduced_set(ON), reduced_set(OFF)
both, on_only, off_only = on & off, on - off, off - on
allset = on | off
print(f"ON={len(on)} OFF={len(off)} both={len(both)} ON-only={len(on_only)} OFF-only={len(off_only)}", flush=True)

# ---- (A) strict-lowering check on every symmetry-solved integral ----
print("\n=== (A) symmetry reductions strictly weight-lowering? ===", flush=True)
sym = sym_solved_set(ON)
bad, raise_w, ok = [], 0, 0
for I in sym:
    sol = R.sol(I)
    if sol is None:        # solved by sym in the worker but R.sol says no -> inconsistency
        bad.append(I); continue
    kI = _target_key(I); wI = _weight(I)[:2]
    for J in sol:
        if _target_key(J) <= kI:            # not strictly lower in the total order
            bad.append((I, J));
        if _weight(J)[:2] > wI:             # higher (r,s) weight
            raise_w += 1
    ok += 1
print(f"  sym-solved checked={len(sym)}  reproduced by R.sol={ok}  "
      f"NOT-strictly-lower violations={len(bad)}  higher-(r,s)-products={raise_w}", flush=True)
if bad[:5]: print("  sample violations:", bad[:5], flush=True)

# ---- (B) are ON-only integrals symmetry-pivots whose products rejoin the tree? ----
def classify(label, S):
    piv = in_tree = orphan = nonpiv = 0
    for I in S:
        sol = R.sol(I)
        if sol is None:
            nonpiv += 1
        else:
            piv += 1
            if all((J in allset) for J in sol): in_tree += 1
            else: orphan += 1
    print(f"  {label}: n={len(S)}  sym-pivot={piv} (products all in tree={in_tree}, "
          f"some new={orphan})  non-pivot={nonpiv}", flush=True)

print("\n=== (B) symmetry footprint in the tree differences ===", flush=True)
classify("ON-only ", on_only)
classify("OFF-only", off_only)
classify("both(sample 2000)", set(list(both)[:2000]))

# ---- orbit-twin test: do ON-only integrals share a symmetry image with the tree? ----
print("\n=== orbit-twin: ON-only integrals whose symmetry image is already reduced ===", flush=True)
twin = 0
for I in on_only:
    sol = R.sol(I)
    if sol and any(J in allset for J in sol):
        twin += 1
print(f"  ON-only that are sym-pivots reducing into the existing tree = {twin} / {len(on_only)}", flush=True)
print("\nDONE", flush=True)
