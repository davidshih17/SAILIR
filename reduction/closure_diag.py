#!/usr/bin/env python
import sys, os, pickle, re
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
red=pickle.load(open("results/meta_reduce/list_TA_reductions.pkl","rb")); REDU=red['reductions']
cache=pickle.load(open("replay/reduction_cache.pkl","rb")); CACHE=cache.get('cache',cache)
rhs=set()
for v in REDU.values(): rhs|=set(v)
MASTERS=rhs-set(REDU)
# the unresolved corners from the failures
for c in [(0,0,1,0,1,1,0,0,0,0,0),(1,0,0,0,1,1,0,0,0,0,0),(1,0,1,0,1,0,0,0,0,0,0),(1,0,1,0,0,1,0,0,0,0,0)]:
    print(f"{c}: in REDU={c in REDU}  in CACHE={c in CACHE}  in MASTERS={c in MASTERS}")
print(f"\nmasters list (75): sample by prop-count")
from collections import Counter
def pc(i): return sum(1 for k in range(8) if i[k]>0)
print(" ", Counter(pc(m) for m in MASTERS))
print(" 2-3 prop masters:", [m for m in MASTERS if pc(m)<=3][:8])
print(f"\nreduction table 'masters' field:", red.get('masters') if 'masters' in red else 'none',)
# is there a fuller reduction anywhere covering low corners?
print("\nother reduction pkls:")
os.system("ls -la results/meta_reduce/ 2>/dev/null | head")
