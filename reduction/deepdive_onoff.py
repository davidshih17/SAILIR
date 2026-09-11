#!/usr/bin/env python
"""DEEP DIVE: why does ON (symmetry) reduce MORE distinct integrals than OFF?
Compares the actual SETS of integrals each run reduced (from result-pkl
filenames), characterizes the symmetric difference by weight / #props, and
cross-references which ON workers symmetry solved. No theory — just the data.
Output is unbuffered; run in background, read the log."""
import glob, os, sys
from collections import Counter

BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
ON  = f"{BASE}/results/probetop84_on"
OFF = f"{BASE}/results/probetop84_off"

def parse_int(path):
    b = os.path.basename(path)
    for ext in (".pkl", ".out", ".log", ".err", ".sub"):
        if b.endswith(ext): b = b[:-len(ext)]
    if not b.startswith("async_"): return None
    toks = b.split("_")
    try:
        ints = tuple(int(x) for x in toks[2:13])
    except ValueError:
        return None
    return ints if len(ints) == 11 else None

def weight(it): return (sum(x for x in it if x > 0), sum(-x for x in it if x < 0))
def sector(it): return tuple(1 if x >= 1 else 0 for x in it[:8])
def nprops(it): return sum(sector(it))

def reduced_set(d):
    s = set()
    for f in glob.glob(f"{d}/work/results/*.pkl"):
        it = parse_int(f)
        if it: s.add(it)
    return s

def sym_solved_set(d):
    s = set()
    for f in glob.glob(f"{d}/work/logs/*.out"):
        it = parse_int(f)
        if not it: continue
        try:
            with open(f, errors="ignore") as fh:
                if any("drained after symmetry" in ln for ln in fh):
                    s.add(it)
        except OSError:
            pass
    return s

def redispatch_count(d):
    """integral -> number of .out logs (dispatch attempts)."""
    c = Counter()
    for f in glob.glob(f"{d}/work/logs/*.out"):
        it = parse_int(f)
        if it: c[it] += 1
    return c

print("=== reduced-integral sets ===", flush=True)
on, off = reduced_set(ON), reduced_set(OFF)
both, on_only, off_only = on & off, on - off, off - on
print(f"ON reduced={len(on)}  OFF reduced={len(off)}  both={len(both)}  "
      f"ON-only={len(on_only)}  OFF-only={len(off_only)}", flush=True)

print("\n=== ON-only integrals by weight (top 20) ===", flush=True)
for w, n in Counter(weight(i) for i in on_only).most_common(20): print(f"   w={w}: {n}", flush=True)
print("=== OFF-only integrals by weight (top 20) ===", flush=True)
for w, n in Counter(weight(i) for i in off_only).most_common(20): print(f"   w={w}: {n}", flush=True)

print("\n=== ON-only by #props ===", dict(sorted(Counter(nprops(i) for i in on_only).items())), flush=True)
print("=== OFF-only by #props ===", dict(sorted(Counter(nprops(i) for i in off_only).items())), flush=True)

print("\n=== symmetry-solved (ON) cross-reference ===", flush=True)
sym = sym_solved_set(ON)
print(f"ON symmetry-solved integrals          = {len(sym)}", flush=True)
print(f"  ...also reduced by OFF (via model)  = {len(sym & off)}", flush=True)
print(f"  ...NOT reduced by OFF at all        = {len(sym - off)}", flush=True)
print(f"  ...that are themselves ON-only      = {len(sym & on_only)}", flush=True)

print("\n=== re-dispatch (same integral dispatched >1 time) ===", flush=True)
rc_on, rc_off = redispatch_count(ON), redispatch_count(OFF)
multi_on  = {k: v for k, v in rc_on.items()  if v > 1}
multi_off = {k: v for k, v in rc_off.items() if v > 1}
print(f"ON : {len(multi_on)} integrals dispatched >1x (max {max(rc_on.values()) if rc_on else 0})", flush=True)
print(f"OFF: {len(multi_off)} integrals dispatched >1x (max {max(rc_off.values()) if rc_off else 0})", flush=True)

print("\n=== sample ON-only integrals (were they symmetry-solved? re-dispatched?) ===", flush=True)
for it in sorted(on_only)[:20]:
    print(f"   {it}  w={weight(it)} nprops={nprops(it)}  "
          f"sym_solved={'Y' if it in sym else 'n'}  dispatches={rc_on.get(it,0)}", flush=True)
print("\n=== sample OFF-only integrals ===", flush=True)
for it in sorted(off_only)[:20]:
    print(f"   {it}  w={weight(it)} nprops={nprops(it)}  dispatches={rc_off.get(it,0)}", flush=True)
print("\nDONE", flush=True)
