#!/usr/bin/env python
"""GRAVITY (GR) canonicalize provider — API parity with canonicalize.py
(_transforms, image_unsigned, _reduce, P, N) so symmetry_route / sector_canon_maps
/ canonical_masters work unchanged via topo_config.canonicalize_module().

Transforms come from results/gr_transforms.pkl (symmetry_engine_GR.py: every Kira
record realized by a verified momentum map + external involutions, den behavior
gated at two evaluation points). Format identical to pentagonbox: per source
sector a list of (M, c), M[i] = {j: coeff}, c[i] = const, all mod P.

ONE BEHAVIORAL DIFFERENCE vs the pentagonbox image function, forced by GR's
LINEAR (eikonal) denominators: a denominator can map to coeff * D_j with
coeff != 1 (a reflection). Such FLIP maps are NOT value identities: eikonal
propagators carry an i-epsilon prescription the rational-function algebra
cannot see (1/(-2k.u + ie) != -1/(2k.u + ie)). EMPIRICALLY LOCKED
(analysis/test_merge_signs_vs_fire.py, 2026-07-17): across 677 FIRE-table
symmetric-pair comparisons, maps whose den rows are all exactly +1 give
unanimously consistent identities (sign +1 on all 22 master pairs); maps with
any den-row flip give contradictory signs — and applying them as identities
manufactures false zeros (I = -I) for integrals FIRE and Kira both keep as
masters.

THEREFORE a denominator row is applicable ONLY with coefficient exactly +1, no
constant part, landing in a DENOMINATOR slot — otherwise the transform is
inapplicable to that integral (returns None; skipping a symmetry is always
safe). Numerator rows keep full coefficients (polynomial factors, no
prescription issue — validated by the same 677 comparisons). Kira's det sign
remains unapplied (Jacobian bookkeeping, not a value-level sign).
"""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir.symmetries import sector_of

# Prime and store are overridable TOGETHER (the assert below enforces that they
# agree). The p101 model needs symmetry transforms at 101: a transform's
# coefficients are rationals reduced mod p and must combine with IBP
# coefficients at the SAME prime, which is why hierarchical_reduction refuses
# --use-symmetry when --prime disagrees with the store's prod_point.
# Defaults unchanged -> an unset environment is the validated 1009 behaviour.
P = int(os.environ.get('SAILIR_SYM_PRIME', 1009))
N = 15
N_DEN = 10

_PKL = os.path.join(ROOT, os.environ.get("SAILIR_SYM_STORE",
                                          "results/gr_transforms.pkl"))
with open(_PKL, "rb") as f:
    _STORE = pickle.load(f)
assert _STORE.get("prod_point", (P,))[0] == P, "gr_transforms.pkl prime mismatch"
_SRC = sorted(_STORE["by_sector"].items())


def _transforms(K):
    """All transforms applicable by super-sector: source sector S covers K's
    sector sK when (S & sK) == sK (same convention as pentagonbox)."""
    sK = sector_of(K)
    for S, mcs in _SRC:
        if (S & sK) == sK:
            for mc in mcs:
                yield mc


def image_unsigned(a, M, c):
    """Value image of integral `a` under transform (M, c); None if inapplicable.
    Den rows: single-term, coefficient EXACTLY +1, no constant, den-slot target
    (the empirically-locked clean-den convention — see module docstring; a flip
    row makes the whole transform inapplicable). Numerator rows: full
    multinomial expansion with coefficients (identical to pentagonbox). Kira
    det sign NOT applied."""
    p = P
    base = [0] * N
    num = []
    for i, ai in enumerate(a):
        if ai > 0:
            row = M.get(i)
            if not row or len(row) != 1 or c.get(i, 0) % p:
                return None
            j, co = next(iter(row.items()))
            if j >= N_DEN or co % p != 1:
                return None                    # ISP-landing or FLIP den: skip
            base[j] += ai
        elif ai < 0:
            row = M.get(i, {})
            if not row and not c.get(i, 0):
                return None
            num.append((-ai, row, c.get(i, 0)))
    res = {tuple(base): 1}
    for power, row, const in num:
        for _ in range(power):
            new = {}
            for integ, co in res.items():
                for j, mij in row.items():
                    ni = list(integ); ni[j] -= 1; ni = tuple(ni)
                    new[ni] = (new.get(ni, 0) + co * mij) % p
                if const:
                    new[integ] = (new.get(integ, 0) + co * const) % p
            res = {k: v for k, v in new.items() if v % p}
    return res


def _reduce(d, rules):
    out = {}
    for v, co in d.items():
        t = rules.get(v)
        if t is not None:
            for w, cw in t.items():
                out[w] = (out.get(w, 0) + co * cw) % P
        else:
            out[v] = (out.get(v, 0) + co) % P
    return {k: x for k, x in out.items() if x % P}


if __name__ == "__main__":
    # smoke gates: transform counts; corner self-consistency. Under the
    # clean-den convention a transform with a FLIP row on the source corner is
    # EXPECTED to be inapplicable (skipped) — the gate requires only that every
    # APPLICABLE transform yields a single-term coefficient-1 corner image.
    n_maps = sum(len(v) for _, v in _SRC)
    print(f"transform store: {len(_SRC)} sectors, {n_maps} transforms")
    bad = clean = flip_skipped = 0
    for S, mcs in _SRC:
        corner = tuple(1 if S >> i & 1 else 0 for i in range(N_DEN)) + (0,) * (N - N_DEN)
        for (M, c) in mcs:
            img = image_unsigned(corner, M, c)
            if img is None:
                flip_skipped += 1
                continue
            clean += 1
            if len(img) != 1 or next(iter(img.values())) % P != 1:
                bad += 1
                print(f"  corner of {S}: bad image {img}")
    print(f"corner applications: clean {clean}, flip-skipped {flip_skipped}, bad {bad}")
    print("ALL PASS" if bad == 0 and clean > 0 else "FAIL")
