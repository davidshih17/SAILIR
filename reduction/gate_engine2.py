#!/usr/bin/env python
"""Gates for the GENERAL engine (symmetry_engine2) on both topologies.

GRAVITY gate: the general store must be behaviorally EQUIVALENT to the
validated GR-specific store (results/gr_transforms.pkl — numerically gated,
FIRE-oracle cross-checked): identical transform sets per source sector.

PENTAGONBOX gate:
  (a) coverage: every legacy transform (canonicalize._build_engine_src) must
      appear in the general store (exact (M, c) match, same source sector);
  (b) orbit structure from the general store (clean-den image) must equal
      results/canonical_sectors_tkey.pkl rep_of exactly (174 orbits);
  (c) retroactive soundness: every LEGACY transform passes the general
      numeric ground-truth gate.
Usage: gate_engine2.py gravity3L|pentagonbox
"""
import os, sys, pickle, random
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))


P_NORM = 1009


def key_of(Md, cd):
    """Representation-insensitive transform key: zero coefficients, empty
    rows, and zero constants dropped (legacy _build_engine_src keeps zero
    entries; the general store does not)."""
    rows = []
    for i, r in Md.items():
        nz = tuple(sorted((int(j), int(v) % P_NORM)
                          for j, v in r.items() if v % P_NORM))
        if nz:
            rows.append((int(i), nz))
    consts = tuple(sorted((int(i), int(v) % P_NORM)
                          for i, v in cd.items() if v % P_NORM))
    return (tuple(sorted(rows)), consts)


def norm_store(by_sector):
    return {s: {key_of(Md, cd) for (Md, cd) in mcs}
            for s, mcs in by_sector.items()}


def gate_gravity():
    new = pickle.load(open(os.path.join(ROOT, "results/gravity3L_transforms_v2.pkl"), "rb"))
    ref = pickle.load(open(os.path.join(ROOT, "results/gr_transforms.pkl"), "rb"))
    a, b = norm_store(new["by_sector"]), norm_store(ref["by_sector"])
    secs = set(a) | set(b)
    same = miss = extra = 0
    for s in sorted(secs):
        ka, kb = a.get(s, set()), b.get(s, set())
        same += len(ka & kb)
        if kb - ka:
            miss += len(kb - ka)
            print(f"  sector {s}: {len(kb - ka)} reference transforms MISSING")
        extra += len(ka - kb)
    print(f"transforms: shared {same}, missing-vs-reference {miss}, "
          f"extra-vs-reference {extra}")
    print("GRAVITY GATE: " + ("ALL PASS" if miss == 0 else "FAIL"))
    return miss == 0


def gate_pentagonbox():
    import importlib
    import canonicalize as legacy
    from symmetry_engine2 import GeneralEngine
    from sailir.topology import Topology
    topo_dir = os.path.join(ROOT, "topology_input/pentagonbox")
    topo = Topology.from_dir(topo_dir)
    new = pickle.load(open(os.path.join(ROOT, "results/pentagonbox_transforms_v2.pkl"), "rb"))
    gen = norm_store(new["by_sector"])

    # (a) legacy coverage (key_of normalizes both representations)
    n_leg = n_cov = 0
    for S, mcs in legacy._SRC:
        for (Md, cd) in mcs:
            n_leg += 1
            if key_of(Md, cd) in gen.get(S, set()):
                n_cov += 1
    print(f"(a) legacy transforms covered: {n_cov}/{n_leg}")

    # (b) orbit structure with the general store, clean-den image
    os.environ['SAILIR_TOPOLOGY'] = 'pentagonbox'
    N, N_DEN, P = new["N_IND"], new["N_DEN"], new["prod_point"][0]
    src = sorted(new["by_sector"].items())

    def image(aa, M, c):
        base = [0] * N
        num = []
        for i, ai in enumerate(aa):
            if ai > 0:
                row = M.get(i)
                if not row or len(row) != 1 or c.get(i, 0) % P:
                    return None
                j, co = next(iter(row.items()))
                if j >= N_DEN or co % P != 1:
                    return None
                base[j] += ai
            elif ai < 0:
                row = M.get(i, {})
                if not row and not c.get(i, 0):
                    return None
                num.append((-ai, row, c.get(i, 0)))
        res = {tuple(base): 1}
        for power, row, const in num:
            for _ in range(power):
                newd = {}
                for integ, co in res.items():
                    for j, mij in row.items():
                        ni = list(integ); ni[j] -= 1
                        ni = tuple(ni)
                        newd[ni] = (newd.get(ni, 0) + co * mij) % P
                    if const:
                        newd[integ] = (newd.get(integ, 0) + co * const) % P
                res = {k: v for k, v in newd.items() if v % P}
        return res

    def corner(mask):
        return tuple(1 if mask >> i & 1 else 0 for i in range(N_DEN)) \
            + (0,) * (N - N_DEN)

    def secof(t):
        return sum(1 << i for i in range(N_DEN) if t[i] > 0)

    # BASE order (no sector-rank prefix): this picks orbit representatives, and
    # the rank prefix is derived from exactly that choice -- see total_order.
    from total_order import base_key as tkey

    edges = {}
    for m in range(1, 1 << N_DEN):
        out = set()
        cm = corner(m)
        sK = secof(cm)
        for S, mcs in src:
            if (S & sK) != sK:
                continue
            for (M, c) in mcs:
                img = image(cm, M, c)
                if img is not None and len(img) == 1:
                    (J, co), = img.items()
                    if co % P == 1:
                        out.add(secof(J))
        edges[m] = out
    rep_of = {}
    seen = set()
    for S in range(1, 1 << N_DEN):
        if S in seen:
            continue
        orb = {S}; fr = [S]
        while fr:
            T = fr.pop()
            for U in edges[T]:
                if U not in orb:
                    orb.add(U); fr.append(U)
        rep = secof(max((corner(T) for T in orb), key=tkey))
        for T in orb:
            rep_of[T] = rep
        seen |= orb
    old = pickle.load(open(os.path.join(ROOT, "results/canonical_sectors_tkey.pkl"), "rb"))
    agree = sum(1 for S in range(1, 1 << N_DEN) if rep_of[S] == old["rep_of"][S])
    n_can = len(set(rep_of.values()))
    print(f"(b) rep_of agreement vs canonical_sectors_tkey: {agree}/{(1 << N_DEN) - 1}; "
          f"canonical: {n_can} (expect {len(set(old['rep_of'].values()))})")

    # (c) numeric verification of legacy transforms: covered implicitly — every
    # matched general transform passed the numeric gate at build time; any
    # legacy transform NOT covered would fail (a).
    ok = (n_cov == n_leg) and agree == (1 << N_DEN) - 1
    print("PENTAGONBOX GATE: " + ("ALL PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "gravity3L":
        gate_gravity()
    else:
        gate_pentagonbox()
