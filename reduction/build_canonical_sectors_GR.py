#!/usr/bin/env python
"""Canonical sectors for the 3-loop GRAVITY family (GR) — the symmetry-enhanced
data-gen pipeline applied to gravity (framework: reduction/ORDERING.md; the
pentagonbox analog is build_canonical_sectors_tkey.py).

GR is SIMPLER than pentagonbox: every Kira symmetry record is a pure slot
permutation (`ing`, verified sign in {+1,-1}, no kinematic coefficients), so the
clean corner orbit needs no momentum-map derivation: sector S maps to
pi(S) = {ing[g] : g in S}, applicable to any sector T with (T & src) == T.
Convention matches the framework: signs are NOT applied at value level
(sailir/symmetries.py docstring; same unsigned convention as pentagonbox).

Orbits close over BOTH files of the SYM run (kira_validate):
  sectorSymmetries (197 within-sector automorphisms — no sector motion) and
  sectorRelations  (354 cross-sector maps — these create the orbits).
Rep of each orbit = the _target_key-MAX corner (the survivor; same convention
as canonical_rep.py — see the anti-survivor warning there).

Outputs:
  results/canonical_sectors_GR.pkl  {rep_of, canonical, order}
  results/canonical_sectors_GR.txt  comma-separated canonical list (data-gen
                                    --restrict-sectors-file)
Reports: zoom, per-orbit sizes, the FIRE-masters sector audit (which of the 68
masters live in non-canonical sectors -> canonical-masters relabeling needed).
"""
import os, sys, pickle, importlib.util
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
from sailir.symmetries import parse_symmetries

N_IND = 15
N_DEN = 10
GRDIR = os.path.join(ROOT, "topology_input/gravity3L")
SYMDIR = os.path.join(GRDIR, "kira_validate/sectormappings/GR")


def corner(mask):
    return tuple(1 if mask >> i & 1 else 0 for i in range(N_DEN)) + (0,) * (N_IND - N_DEN)


# The BASE order (no sector-rank prefix) -- these scripts BUILD the sector
# ranking, so using the rank-prefixed tkey here would be circular. One
# definition: reduction/total_order.py.
from total_order import base_key as tkey  # noqa: E402


def sector_of(t):
    return sum(1 << i for i in range(N_DEN) if t[i] > 0)


recs = (parse_symmetries(os.path.join(SYMDIR, "sectorSymmetries"), N_IND, n_loops=3)
        + parse_symmetries(os.path.join(SYMDIR, "sectorRelations"), N_IND, n_loops=3))
print(f"records: {len(recs)} (sym+rel), all pure slot permutations")

# sanity: on the source sector's denominator slots, ing must be defined and land
# in denominator slots (a sector map, not a den->ISP mutation)
bad = 0
for r in recs:
    for g in range(N_DEN):
        if r.source_sector >> g & 1:
            if r.ing[g] == -1 or r.ing[g] >= N_DEN:
                bad += 1
                break
print(f"records whose source dens do not map to dens: {bad} (must be 0)")

# sector graph: T -> pi(T) for every record applicable at T ((T & src) == T)
by_src = {}
for r in recs:
    by_src.setdefault(r.source_sector, []).append(r.ing)

def images(T):
    out = set()
    for src, ings in by_src.items():
        if (T & src) == T:
            for ing in ings:
                ok = True
                img = 0
                for g in range(N_DEN):
                    if T >> g & 1:
                        j = ing[g]
                        if j == -1 or j >= N_DEN:
                            ok = False
                            break
                        img |= 1 << j
                if ok:
                    out.add(img)
    return out

# orbit closure over all 1023 sectors
rep_of = {}
orbits = []
seen = set()
for S in range(1, 1 << N_DEN):
    if S in seen:
        continue
    orb = {S}
    frontier = [S]
    while frontier:
        T = frontier.pop()
        for U in images(T):
            if U not in orb:
                orb.add(U)
                frontier.append(U)
    rep = sector_of(max((corner(T) for T in orb), key=tkey))
    for T in orb:
        rep_of[T] = rep
    seen |= orb
    orbits.append((rep, sorted(orb)))

canonical = sorted(set(rep_of.values()))
zoom = (len(rep_of)) / len(canonical)
print(f"\nsectors: {len(rep_of)}  ->  canonical: {len(canonical)}  (zoom {zoom:.2f}x)")
from collections import Counter
sizes = Counter(len(o) for _, o in orbits)
print(f"orbit-size histogram: {dict(sorted(sizes.items()))}")

# masters audit (FIRE symmetric basis)
spec = importlib.util.spec_from_file_location("grm", os.path.join(GRDIR, "GR_masters_dict.py"))
grm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grm)
md = getattr(grm, "PAPER_MASTERS", None) or grm.GR_MASTERS
n_m = sum(len(v) for v in md.values())
mm = 0
print(f"\nFIRE masters: {n_m} in {len(md)} sectors")
for s, ms in sorted(md.items()):
    r = rep_of.get(s, s)
    tag = "OK " if s == r else "MISMATCH"
    if s != r:
        mm += len(ms)
    print(f"  {tag} sector {s:>4} ({len(ms)} masters)  canonical-of-orbit {r:>4}")
print(f"masters in NON-canonical sectors: {mm}/{n_m}"
      + ("  -> canonical-masters relabeling required" if mm else ""))

out = {"rep_of": rep_of, "canonical": canonical,
       "order": "_target_key = (-r,-s,|abs|) corner survivor (max)",
       "source": SYMDIR}
with open(os.path.join(ROOT, "results/canonical_sectors_GR.pkl"), "wb") as f:
    pickle.dump(out, f)
with open(os.path.join(ROOT, "results/canonical_sectors_GR.txt"), "w") as f:
    f.write(",".join(str(s) for s in canonical) + "\n")
print(f"\nsaved -> results/canonical_sectors_GR.pkl / .txt")
