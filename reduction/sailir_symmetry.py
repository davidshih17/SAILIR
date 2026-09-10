#!/usr/bin/env python
"""Greedy sector-symmetry reduction for the v7 beam search (inference-only).

A sector symmetry is a deterministic linear relation I = sum_k c_k J_k. We orient
it by SAILIR's own total order (_target_key) to eliminate the integral SAILIR
would target, producing a `sol = {J_k: c_k mod p}` in the SAME format
solve_ibp_for returns — so it drops straight into apply_substitution_v5.

Coefficients are evaluated at SAILIR's fixed kinematic point (mod p), so they are
consistent with the IBP coefficients. Confluent: any applicable symmetry reduces
toward the same canonical rep, so we just take the first that fires.
"""
import os, sys, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))
from sailir.symmetries import parse_symmetries, sector_of
from symmetry_engine import derive_transform, N

TA_SYMDIR = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
_CACHE = os.path.join(ROOT, "reduction", "sailir_symmetry_transforms.pkl")


def _weight(i):
    return (sum(max(0, x) for x in i), -sum(min(0, x) for x in i), tuple(abs(x) for x in i))

# Total order: ONE definition, reduction/total_order.py. Re-implementing it
# here is how canonical_rep drifted to a range(8) sector mask and disagreed
# with the live order on 69% of gravity3L integrals.
from total_order import tkey as _target_key  # noqa: E402


def _eval_const(s, kin, prime):
    if s in ("0", ""):
        return 0
    return eval(s.replace("^", "**"), {"__builtins__": {}}, dict(kin)) % prime


def _parse_raw_matrices(prime, kin):
    """within-sector transforms from Kira's raw 'symmetries' matrices ->
    by_source[sec] += [(M, c)], M[i]={j:int}, c[i]=int mod p."""
    blocks, cur = [], []
    for ln in open(os.path.join(TA_SYMDIR, "symmetries")):
        s = ln.rstrip("\n")
        if not s.strip():
            if cur:
                blocks.append(cur); cur = []
            continue
        cur.append(s)
    if cur:
        blocks.append(cur)
    out = {}
    for b in blocks:
        src = int(b[1].split()[1])
        M, c = {}, {}
        for i, row in enumerate(b[3:3 + N]):
            t = row.split()
            coeffs, const = t[:N], (t[N] if len(t) > N else "0")
            M[i] = {j: int(coeffs[j]) % prime for j in range(N) if coeffs[j] != "0"}
            c[i] = _eval_const(const, kin, prime)
        out.setdefault(src, []).append((M, c))
    return out


def _derive_cross(prime, kin):
    """cross-sector transforms from sectorRelations via the momentum engine,
    evaluated at the kinematic point mod p."""
    import sympy as sp
    syms = {sp.Symbol(k): v for k, v in kin.items()}
    rels = parse_symmetries(os.path.join(TA_SYMDIR, "sectorRelations"), N, 2)
    out = {}
    for r in rels:
        if any(rhs == "placeholder" for _, rhs in r.loop_substs):
            # placeholder: denominator permutation only (ing), no numerator info
            M = {g: {t: 1} for g, t in enumerate(r.ing) if t != -1}
            c = {g: 0 for g in range(N)}
        else:
            Mf, cf = derive_transform(list(r.loop_substs))
            M = {i: {j: int(coef.subs(syms)) % prime for j, coef in Mf[i].items()} for i in Mf}
            c = {i: int(cf[i].subs(syms)) % prime for i in cf}
        out.setdefault(r.source_sector, []).append((M, c))
    return out


class SymmetryReducer:
    def __init__(self, prime=1009, kin=None):
        self.prime = prime
        self._cache = {}            # memoize sol() per integral (None = irreducible)
        self.kin = kin or {"d": 41, "s12": 31, "s23": 47, "s34": 53, "s45": 59, "s51": 61}
        key = (prime, tuple(sorted(self.kin.items())))
        loaded = None
        if os.path.exists(_CACHE):
            with open(_CACHE, "rb") as f:
                d = pickle.load(f)
            if d.get("key") == key:
                loaded = d["by_source"]
        if loaded is None:
            by_source = _parse_raw_matrices(prime, self.kin)
            for src, lst in _derive_cross(prime, self.kin).items():
                by_source.setdefault(src, []).extend(lst)
            with open(_CACHE, "wb") as f:
                pickle.dump({"key": key, "by_source": by_source}, f)
            loaded = by_source
        self.by_source = loaded

    def _image(self, a, M, c):
        """I = sum_k coeff_k J_k, returned as {J: coeff mod p}. Denominators map
        cleanly (single +/-1); numerators expand (multinomial)."""
        p = self.prime
        base = [0] * N
        sign = 1
        num_factors = []
        for i, ai in enumerate(a):
            if ai > 0:
                row = M.get(i)
                if not row or len(row) != 1:
                    return None                      # denominator -> combination: inapplicable
                j, co = next(iter(row.items()))
                base[j] += ai
                if co % p == p - 1 and ai % 2 == 1:  # co == -1 and odd power
                    sign = (-sign) % p
            elif ai < 0:
                row = M.get(i, {})
                if not row and not c.get(i, 0):
                    return None
                num_factors.append((-ai, row, c.get(i, 0)))
        result = {tuple(base): sign % p}
        for power, row, const in num_factors:
            for _ in range(power):
                new = {}
                for integ, co in result.items():
                    for j, mij in row.items():
                        ni = list(integ); ni[j] -= 1; ni = tuple(ni)
                        new[ni] = (new.get(ni, 0) + co * mij) % p
                    if const:
                        new[integ] = (new.get(integ, 0) + co * const) % p
                result = {k: v for k, v in new.items() if v % p != 0}
        return result

    def sol(self, I):
        """If I is the _target_key-minimum (SAILIR's pivot) of some symmetry image,
        return sol = {J: c} with I = sum_J c_J J (mod p); else None. Memoized."""
        c = self._cache
        if I in c:
            return c[I]
        r = self._sol(I)
        c[I] = r
        return r

    def _sol(self, I):
        p = self.prime
        kI = _target_key(I)
        for (M, c) in self.by_source.get(sector_of(I), []):
            img = self._image(I, M, c)
            if img is None:
                continue
            others = {J: co for J, co in img.items() if J != I}
            if not others:
                continue
            if all(_target_key(J) > kI for J in others):   # I is the pivot to eliminate
                selfco = img.get(I, 0) % p
                denom = (1 - selfco) % p
                if denom == 0:
                    continue                                # can't solve for I (degenerate)
                inv = pow(denom, p - 2, p)
                return {J: (co * inv) % p for J, co in others.items()}
        return None
