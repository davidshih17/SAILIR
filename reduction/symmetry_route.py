#!/usr/bin/env python
"""Design 1 routing layer: given an integral I, return its STRICTLY-LOWERING
symmetry rewrite  I -> {lower individual integrals}  (value-preserving), or None
if I is a symmetry survivor (must go to an IBP worker).

Mechanism (reuses canonicalize's Kira-validated transforms):
  * union-close {I} under every applicable (super-sector) symmetry, forming exact
    value relations  K - image(K) = 0  (image via image_unsigned, no det sign);
  * RREF pivoting on HIGHEST total-weight. Symmetry never raises weight, so I is
    the max-weight element of its closure => if I is reducible it is the pivot and
    its rule RHS is automatically strictly-lower total-weight.
  * symmetry_rule(I) returns rules.get(I): {} means I is a symmetry-zero (I->0),
    a non-empty dict is the strictly-lower rewrite, None means survivor.

Every rule is a GF(p) combination of exact symmetry identities => the master
combination is unchanged (verified by the A/B masters-equality gate).
"""
import os, sys
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
# topology-keyed provider (SAILIR_TOPOLOGY, default pentagonbox = the original
# 'canonicalize' momentum-engine module — behavior unchanged with the flag unset)
import topo_config as _tc
_cz = _tc.canonicalize_module()
_transforms, image_unsigned, _reduce, P = (_cz._transforms, _cz.image_unsigned,
                                           _cz._reduce, _cz.P)
_N_DEN, _N_ISP = _tc.N_DEN, _tc.N_IND - _tc.N_DEN

# SAILIR_SECTOR_RANK=1 -> the adopted sector-senior order (ORDERING.md): sector rank
# first, then (r,s), then |abs|. Must be set identically for the orchestrator AND all
# workers of a run (one shared order = confluence). Default 0 = legacy order.
_SECTOR_RANK = os.environ.get('SAILIR_SECTOR_RANK', '0') == '1'
if _SECTOR_RANK:
    from sector_rank import RANK_IDX as _RANK_IDX

# NOTE on scaleless (trivial-sector) integrals: a symmetry relabeling can map a
# nonzero integral onto a scaleless sub-sector integral (= 0), e.g. D1 D5 D6 ->
# D5 D6 D8 (k1 only in (k1-k2)^2). We do NOT special-case these: under
# --paper-masters-only they are not masters, so a scaleless survivor is dispatched
# to an IBP worker which reduces it to 0 the SAME way the baseline does (verified:
# worker(I[0,0,0,0,1,1,0,1]) -> {}). Symmetry alone can't kill a scaleless integral
# (that needs the scaling symmetry / IBP), so we let IBP finish the job -- no
# explicit zero-sector list. (Under --no-paper-masters-only scaleless corners are
# wrongly treated as masters and the baseline only survives via mod-p cancellation,
# which the symmetry merge breaks -- so ALWAYS use --paper-masters-only here.)


def tkey(i):
    """The WORKERS' total order = beam_search_v7._target_key = (-r, -s, |abs|).
    SMALLER tkey == HIGHER in the ordering == the integral IBP eliminates first.
    A valid reduction sends an integral to terms with strictly LARGER tkey (lower
    in the ordering). symmetry_route MUST use this exact order, or its rewrites run
    opposite to the workers on the |abs| tiebreak and the mixed cache cycles.

    DESIGN DECISION 2026-07-10 (see reduction/ORDERING.md): the adopted order is
    SECTOR RANK first, then (r,s), then |abs| — it gives the HARD guarantee that
    only canonical sectors are ever dispatched (legacy order leaks 2.6-15% of
    dispatches into non-canonical sectors via the |abs| tiebreak). This function
    still implements the LEGACY order; migrate JOINTLY with beam_search_v7,
    canonical_rep, and data-gen at the retrain — never alone.
    SAILIR_SECTOR_RANK=1 switches to the adopted order (rank prefix)."""
    base = (-sum(x for x in i if x > 0), -sum(-x for x in i if x < 0),
            tuple(abs(x) for x in i))
    if not _SECTOR_RANK:
        return base
    m = 0
    for k in range(_N_DEN):
        if i[k] > 0:
            m |= 1 << k
    return (-_RANK_IDX[m],) + base


def _build_rules(seeds, cap=20000):
    """Orbit closure of `seeds` under all applicable symmetries, RREF pivoting on
    the HIGHEST-in-ordering integral (smallest tkey) so every rule's RHS is strictly
    LOWER in the ordering (larger tkey) -- same descent direction as the IBP workers,
    which makes the combined cache a DAG (no cycles). Returns {pivot: {lower: coeff}}
    or None if the closure exceeds `cap` (bail -> caller dispatches a worker)."""
    seen = set(seeds); frontier = list(seeds); raw = []
    while frontier:
        K = frontier.pop()
        for (M, c) in _transforms(K):
            img = image_unsigned(K, M, c)
            if img is None:
                continue
            rel = dict(img); rel[K] = (rel.get(K, 0) - 1) % P
            rel = {k: v for k, v in rel.items() if v % P}
            if rel:
                raw.append(rel)
            for J in img:
                if J not in seen:
                    if len(seen) >= cap:
                        return None
                    seen.add(J); frontier.append(J)
    # pos increases as tkey DECREASES (highest-in-ordering -> largest pos), so
    # max(pos) picks the integral the reduction should eliminate.
    pos = {v: k for k, v in enumerate(sorted(seen, key=tkey, reverse=True))}
    rules = {}
    for rel in raw:
        r = _reduce(rel, rules)
        if not r:
            continue
        piv = max(r, key=lambda v: pos[v])                        # eliminate highest-in-ordering
        co = r.pop(piv); inv = pow(co, P - 2, P)
        newp = {k: (-v * inv) % P for k, v in r.items()}
        for q in list(rules):
            if piv in rules[q]:
                cc = rules[q].pop(piv)
                for w, cw in newp.items():
                    rules[q][w] = (rules[q].get(w, 0) + cc * cw) % P
                rules[q] = {k: v for k, v in rules[q].items() if v % P}
        rules[piv] = newp
    return rules


# ---------------------------------------------------------------------------
# DESCENT IS MODULO THE TERMINAL SET.
#
# A routing rule I = sum c_J J is a valid REDUCTION when every RHS term either
# descends in the total order OR is terminal. Terminality is not a tkey
# property: a master or corner needs no further reduction wherever it sits in
# the ordering, so it can never close a worker/symmetry cycle. Testing
# `all(tkey(k) > ki)` alone therefore REJECTS legitimate rules whose only
# offending terms are masters or corners -- routing opportunities silently lost,
# with the integral falling through to a worker that must redo the work.
#
# is_master() is the terminal set the workers actually use: the topology's
# master basis UNION corner integrals in sectors the basis does not cover
# (unless PAPER_MASTERS_ONLY). It requires ibp_env to be initialised; if it is
# not, MASTERS_SET is empty and every term reads as non-terminal, which silently
# reduces this to the old over-conservative test -- hence the loud warning.
_TERMINAL_WARNED = False


def _terminal(k):
    """True if k is a master or corner, i.e. needs no further reduction."""
    global _TERMINAL_WARNED
    from sailir import ibp_env as _ie
    if not _ie.MASTERS_SET and not _TERMINAL_WARNED:
        _TERMINAL_WARNED = True
        import sys as _sys
        print("[symmetry_route] WARNING: MASTERS_SET is EMPTY -- ibp_env was "
              "never initialised in this process, so no term can be recognised "
              "as terminal and routing falls back to the over-conservative "
              "tkey-only descent test. Call init_from_topology() (and "
              "apply_canonical_masters() under SAILIR_SECTOR_RANK=1) first.",
              file=_sys.stderr, flush=True)
    return _ie.is_master(k)


def symmetry_rule(I):
    """Symmetry rewrite for I that is strictly LOWER in the workers' ordering, or
    None if I is a survivor. Return value:
      {}                 -> I is a symmetry-zero (route I -> 0)
      {lower: coeff,...}  -> rewrite into terms strictly lower in the ordering (route free)
      None               -> survivor (dispatch an IBP worker)
    I is routed only when it is the highest-in-ordering member of its orbit closure
    (so it is a pivot) and every RHS term either is strictly lower in the ordering
    -- the same descent direction as the IBP workers, so no worker/symmetry cycle
    forms -- OR is TERMINAL (master/corner), which needs no further reduction and
    so cannot close a cycle either.
    """
    I = tuple(I)
    rules = _build_rules({I})
    if rules is None or I not in rules:
        return None
    tail = rules[I]
    ki = tkey(I)
    # descend OR be terminal -- see _terminal() above
    if all(tkey(k) > ki or _terminal(k) for k in tail):
        return tail
    return None


# ============================================================================
# STAGED routing (the adopted design, 2026-07-10): two discrete steps instead of
# the monolithic all-symmetry closure above.
#   step 1  sector-changing: non-canonical sector -> ONE precomputed composite map
#           (sector_canon_maps) sends I to canonical-sector terms + lower debris.
#   step 2  within-sector: canonical-sector integrals reduced to the lowest-lex
#           representatives of their WITHIN-sector orbit (sector-preserving maps
#           only), or dropped entirely (the I = -I + lower identities live here).
#   step 3  survivors -> the one-step worker queue (caller).
# REQUIRES the sector-senior order (SAILIR_SECTOR_RANK=1): step 1's rewrite is
# strictly descending only when sector rank is the senior key.
# ============================================================================
_canon_maps = None
_canon_set = None
_within_cache = {}


def _sector_of(i):
    m = 0
    for k in range(_N_DEN):
        if i[k] > 0:
            m |= 1 << k
    return m


def _staged_init():
    global _canon_maps, _canon_set
    if _canon_maps is None:
        assert _SECTOR_RANK, ("staged routing requires SAILIR_SECTOR_RANK=1 "
                              "(step-1 descent is only guaranteed sector-senior)")
        import sector_canon_maps, pickle
        _canon_maps = sector_canon_maps.load()
        with open(_tc.CANON_PKL, "rb") as f:
            _canon_set = set(pickle.load(f)["canonical"])


def _within_transforms(S):
    """Transforms that act WITHIN sector S: applicable at S and mapping the present
    propagators onto themselves (single-prop rows). Cached per sector."""
    if S in _within_cache:
        return _within_cache[S]
    pr = {i for i in range(_N_DEN) if S >> i & 1}
    probe = tuple(1 if i in pr else 0 for i in range(_N_DEN)) + (0,) * _N_ISP
    out = []
    for (M, c) in _transforms(probe):
        img_pr = set()
        ok = True
        for i in pr:
            row = M.get(i, {})
            if len(row) != 1:
                ok = False; break
            (j, _co), = row.items()
            if j >= _N_DEN:
                ok = False; break
            img_pr.add(j)
        if ok and img_pr == pr:
            out.append((M, c))
    _within_cache[S] = out
    return out


def _build_rules_within(I, S, cap=20000):
    """_build_rules restricted to the within-sector maps of S (closure stays in the
    cone of S, relations are within-sector value identities only)."""
    seen = {I}; frontier = [I]; raw = []
    wt = _within_transforms(S)
    while frontier:
        K = frontier.pop()
        for (M, c) in wt:
            img = image_unsigned(K, M, c)
            if img is None:
                continue
            rel = dict(img); rel[K] = (rel.get(K, 0) - 1) % P
            rel = {k: v for k, v in rel.items() if v % P}
            if rel:
                raw.append(rel)
            for J in img:
                if J not in seen:
                    if len(seen) >= cap:
                        return None
                    seen.add(J); frontier.append(J)
    pos = {v: k for k, v in enumerate(sorted(seen, key=tkey, reverse=True))}
    rules = {}
    for rel in raw:
        r = _reduce(rel, rules)
        if not r:
            continue
        piv = max(r, key=lambda v: pos[v])
        co = r.pop(piv); inv = pow(co, P - 2, P)
        newp = {k: (-v * inv) % P for k, v in r.items()}
        for q in list(rules):
            if piv in rules[q]:
                cc = rules[q].pop(piv)
                for w, cw in newp.items():
                    rules[q][w] = (rules[q].get(w, 0) + cc * cw) % P
                rules[q] = {k: v for k, v in rules[q].items() if v % P}
        rules[piv] = newp
    return rules


def canonical_monolithic_rule(I):
    """THE PRODUCTION ROUTER (2026-07-11, supersedes both plain symmetry_rule and
    staged_rule as the default): the monolithic all-symmetry row reduction --
    measured to emit ~1.5x more compact rules than the staged factoring (6.3 vs 9.3
    RHS terms; m2: 158 vs 235 workers) -- PLUS the hard guarantee that only
    CANONICAL-sector integrals are ever emitted as survivors: if the monolithic
    solve leaves a non-canonical-sector integral unresolved (closure cap, exotic
    orbit), fall back to its single composite canonicalization map, which is proven
    strictly descending under the sector-senior order. Requires
    SAILIR_SECTOR_RANK=1. Sector-0 (scaleless debris) integrals remain survivors
    for IBP to dispose of, per the scaleless policy above."""
    I = tuple(I)
    r = symmetry_rule(I)
    if r is not None:
        return r
    _staged_init()
    S = _sector_of(I)
    if S == 0 or S in _canon_set:
        return None
    M, c = _canon_maps[S]
    img = image_unsigned(I, M, c)
    ki = tkey(I)
    if img is None:
        # The composite map is inapplicable to THIS integral — in practice a
        # POSITIVE ISP power (the ISP as a denominator; a legitimate transient
        # of IBP algebra, cf. pentagonbox's 20 non-ispclean list_TA entries):
        # symmetry rows for ISP slots are combinations, so no clean rewrite
        # exists. Survivor -> the IBP worker reduces it directly. This is the
        # ONE sanctioned relaxation of the only-canonical-sectors guarantee.
        return None
    assert all(tkey(k) > ki or _terminal(k) for k in img), (
        f"canonicalization fallback failed to descend for {list(I)} — "
        f"sector-senior order violated?")
    return img


def staged_rule(I):
    """The staged router. Same contract as symmetry_rule:
      {} / {lower: coeff, ...}  -> free rewrite, strictly lower in the order
      None                      -> survivor (canonical sector, within-orbit lowest)
    Step 1 for non-canonical sectors (one composite map application); step 2 for
    canonical sectors (within-sector orbit RREF). The orchestrator cascade re-offers
    the RHS terms on later rounds, which sequences 1 -> 2 -> queue per integral."""
    I = tuple(I)
    _staged_init()
    S = _sector_of(I)
    ki = tkey(I)
    if S not in _canon_set:
        # step 1: canonicalize the sector with the single composite map
        g = _canon_maps.get(S)
        if g is None:
            # S == 0: no denominators at all (scaleless, appears as cascade debris).
            # Per the scaleless policy above: no special-casing — survivor, the IBP
            # worker reduces it to 0 exactly like the baseline.
            return None
        M, c = g
        img = image_unsigned(I, M, c)
        if img is not None and all(tkey(k) > ki or _terminal(k) for k in img):
            return img
        return None            # cannot happen under sector-senior order; safe fallback
    # step 2: within-sector reduction to the lowest-lex orbit representative
    rules = _build_rules_within(I, S)
    if rules is None or I not in rules:
        return None
    tail = rules[I]
    if all(tkey(k) > ki or _terminal(k) for k in tail):
        return tail
    return None


if __name__ == "__main__":
    from sailir import ibp_env
    from sailir.topology import Topology
    ibp_env.init_from_topology(Topology.from_dir(os.path.join(ROOT, "topology_input/pentagonbox_nosym")))
    ibp_env.set_prime(1009)
    tests = [
        (2, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0),   # D2 present: must NOT route to I[1,1,2] (fake-M bug)
        (2, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0),   # D2 absent: D1^2 D3 -> should route (D1<->D3)
        (1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0),   # a plain 5-prop corner (likely survivor)
        (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0),   # top-sector corner (asymmetric -> survivor)
    ]
    for I in tests:
        r = symmetry_rule(I)
        if r is None:
            print(f"  {I}  ->  SURVIVOR (worker)")
        elif not r:
            print(f"  {I}  ->  0  (symmetry-zero)")
        else:
            k = tkey(I)
            lo = all(tkey(t) > k for t in r)     # lower in ordering (larger tkey)
            terms = " + ".join(f"{c}*{t}" for t, c in list(r.items())[:3])
            print(f"  {I}  ->  {terms}{' + ...' if len(r) > 3 else ''}   [lower-in-ordering={lo}]")
