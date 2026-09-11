#!/usr/bin/env python
"""Is a given integral a 142-basis master, or related to one by the symmetry
maps (canonical image lands on a master / same orbit)? Also: reducible per the
truth engine? Usage: check_master_status_one.py 'i1,i2,...'"""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
import topo_config as _tc

T = tuple(int(x) for x in sys.argv[1].split(","))
masters = set()
for ln in open(os.path.join(ROOT, "results/gr_cut_masters_142.txt")):
    masters.add(tuple(int(x) for x in ln.split(",")))


def sec(t):
    return sum(1 << i for i in range(_tc.N_DEN) if t[i] > 0)


S = sec(T)
r = sum(x for x in T if x > 0)
s = sum(-x for x in T if x < 0)
cm = pickle.load(open(_tc.CANON_PKL, "rb"))
canon, rep = set(cm["canonical"]), cm["rep_of"]
print(f"integral {list(T)}  sector {S} (canonical={S in canon}, "
      f"rep={rep.get(S)})  r={r} s={s}")
print(f"is a 142-basis master: {T in masters}")
sec_masters = [m for m in masters if sec(m) == S]
print(f"masters in its sector: {len(sec_masters)}")
for m in sec_masters:
    print(f"   {list(m)}")

# symmetry image check: canonical image of T (if sector non-canonical) or
# in-orbit relation to any master
CG = _tc.canonicalize_module()
import sector_canon_maps
maps = sector_canon_maps.load()
if S not in canon:
    M, c = maps[S]
    img = CG.image_unsigned(T, M, c)
    print(f"canonical image: {img if img is None else {tuple(k): v for k, v in img.items()}}")
    if img:
        hits = [k for k in img if tuple(k) in masters]
        print(f"image terms that are masters: {[list(k) for k in hits]}")
else:
    print("sector is canonical: symmetry routing leaves it in place")
