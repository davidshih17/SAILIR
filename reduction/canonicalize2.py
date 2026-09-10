#!/usr/bin/env python
"""GENERAL canonicalize provider — API parity with canonicalize.py
(_transforms, image_unsigned, _reduce, P, N) for ANY topology, backed by a
symmetry_engine2 store. Selected via topo_config.CANONICALIZE_MOD =
'canonicalize2'; the store path comes from topo_config.STORE_PKL.

Applies the empirically locked CLEAN-DEN convention (see canonicalize_GR.py /
analysis/test_merge_signs_vs_fire.py): a denominator row is applicable ONLY
single-term with coefficient exactly +1, no constant, landing in a denominator
slot — flip maps are not value identities for eikonal (linear) denominators.
Numerator rows keep full coefficients. Kira's det sign is never applied.
"""
import os
import pickle
import topo_config as _tc

with open(_tc.STORE_PKL, "rb") as f:
    _STORE = pickle.load(f)
P = _STORE["prod_point"][0]
N = _STORE["N_IND"]
N_DEN = _STORE["N_DEN"]
_SRC = sorted(_STORE["by_sector"].items())
assert N == _tc.N_IND and N_DEN == _tc.N_DEN, \
    f"store {_tc.STORE_PKL} does not match topology {_tc.TOPOLOGY}"


def _sector_of(t):
    m = 0
    for i in range(N_DEN):
        if t[i] > 0:
            m |= 1 << i
    return m


def _transforms(K):
    sK = _sector_of(K)
    for S, mcs in _SRC:
        if (S & sK) == sK:
            for mc in mcs:
                yield mc


_MAX_EXPAND = int(os.environ.get('SAILIR_ROUTE_MAX_EXPAND', '0'))


def image_unsigned(a, M, c):
    """Value image of integral `a` under transform (M, c); None if
    inapplicable. CLEAN-DEN convention enforced."""
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
            new = {}
            for integ, co in res.items():
                for j, mij in row.items():
                    ni = list(integ); ni[j] -= 1; ni = tuple(ni)
                    new[ni] = (new.get(ni, 0) + co * mij) % p
                if const:
                    new[integ] = (new.get(integ, 0) + co * const) % p
            res = {k: v for k, v in new.items() if v % p}
            # COMPUTE CAP. A term-count cap applied to the RETURNED rule stops
            # the expression from growing but saves no time -- the giant
            # expansion is still built, then discarded. Measured on the real
            # frontier: an s=20 integral expands to 1,315,600 terms. Bailing
            # here abandons the expansion as soon as it is clearly headed there.
            #
            # None already means "this transform does not apply" and every
            # caller treats it as skip-and-continue, so an over-cap map is
            # simply not used -- the integral falls through to worker dispatch.
            #
            # Deliberately MUCH larger than the returned-rule cap: cancellation
            # in the `% p` filter can shrink res between passes, so a route with
            # a small final rule may spike mid-expansion. A generous ceiling
            # rejects only genuine explosions. 0 disables (exact prior
            # behaviour, which is the default).
            if _MAX_EXPAND and len(res) > _MAX_EXPAND:
                return None
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
