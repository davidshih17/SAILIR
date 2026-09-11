#!/usr/bin/env python
"""How much does CLEAN-orbit canonicalization actually dedup? Scan corners + numerator
integrals across all sectors, compute clean-orbit size, report how many are non-trivial
(orbit>1 => they merge with something). This decides whether canonicalizing the training
data to clean reps is worth it."""
import sys, os, itertools
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from canonical_rep import clean_orbit
ISP=[8,9,10]
def scan(label, gen):
    n=nontriv=0; sizes=[]
    for I in gen:
        orb=clean_orbit(I)
        if orb is None: continue
        n+=1; sizes.append(len(orb))
        if len(orb)>1: nontriv+=1
    import statistics as st
    print(f"  {label:34s}: {n} integrals, {nontriv} with orbit>1 ({100*nontriv/max(1,n):.0f}%), "
          f"orbit size median={st.median(sizes) if sizes else 0:.0f} max={max(sizes) if sizes else 0}")

def corners():
    for mask in range(1,256):
        yield tuple(1 if k in [j for j in range(8) if mask>>j&1] else 0 for k in range(11))
def numer(deg):
    for mask in range(1,256):
        combo=[k for k in range(8) if mask>>k&1]
        corner=tuple(1 if k in combo else 0 for k in range(11))
        for cp in itertools.combinations_with_replacement(ISP,deg):
            I=list(corner)
            for s in cp: I[s]-=1
            yield tuple(I)
print("clean-orbit dedup coverage:")
scan("corners (all sectors)", corners())
scan("deg-1 ISP-numerator integrals", numer(1))
scan("deg-2 ISP-numerator integrals", numer(2))
