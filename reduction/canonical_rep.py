#!/usr/bin/env python
"""Symmetry canonicalization for the training/inference pipeline.

canonical_rep(I) maps an integral to the SURVIVOR of its clean-permutation orbit --
the member the reduction keeps -- using the WORKERS' total order
`_target_key = (-r,-s,|abs|)`. The reduction eliminates the SMALLEST `_target_key`
first (highest in the ordering) and sends it to strictly-LARGER `_target_key` terms,
so the survivor is the `_target_key`-MAXIMAL member (== the fixed point `symmetry_route`
routes every other orbit member to). Two integrals related by a clean loop relabeling
(value-equal single integrals) map to the same rep; the affine relations that produce
combinations are NOT used here (they are the action-space relations, not renames).
This is the dedup canonicalization -- 1->1, no P_S expansion.

CRITICAL: the order is `_target_key`, matching beam_search_v7 and symmetry_route, so
the canonical basis is confluent with the reduction. (The old min-integer `rep_of` /
`canon()` / 139-sector treatment has been removed from the codebase — only this 174
`_target_key` clean-orbit treatment, the one the successful symmetry inference uses,
remains.)
"""
import os, sys
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))
from canonicalize import _transforms, image_unsigned, P

# SAILIR_SECTOR_RANK=1 -> adopted sector-senior order (ORDERING.md). Must match
# symmetry_route and the workers for the run. Default 0 = legacy.
_SECTOR_RANK = os.environ.get('SAILIR_SECTOR_RANK', '0') == '1'
if _SECTOR_RANK:
    from sector_rank import RANK_IDX as _RANK_IDX


# Total order: ONE definition, reduction/total_order.py. Re-implementing it
# here is how canonical_rep drifted to a range(8) sector mask and disagreed
# with the live order on 69% of gravity3L integrals.
from total_order import tkey as tkey  # noqa: E402


def clean_orbit(I, cap=4000):
    """Set of integrals reachable from I by CLEAN (single-integral, value-equal)
    symmetry relabelings. Bounded by `cap`."""
    I = tuple(I)
    seen = {I}; frontier = [I]
    while frontier:
        K = frontier.pop()
        for (M, c) in _transforms(K):
            img = image_unsigned(K, M, c)
            if img is None or len(img) != 1:
                continue                          # not a clean rename -> skip (affine relation)
            (J, co), = img.items()
            if co % P != 1:                       # value relation must be +1 (no spurious sign)
                continue
            if J not in seen:
                if len(seen) >= cap:
                    return None                   # runaway -> bail (treat I as its own rep)
                seen.add(J); frontier.append(J)
    return seen


def canonical_rep(I):
    """The `_target_key`-MAXIMAL member of I's clean orbit = the SURVIVOR the reduction
    keeps (symmetry_route routes every other orbit member TO this one). The reduction
    eliminates the SMALLEST _target_key first and sends it to strictly-LARGER _target_key
    (lower-weight) terms, so the surviving canonical form is the MAXIMUM, not the minimum.
    Idempotent. Falls back to I itself if the orbit is a singleton or exceeds cap."""
    I = tuple(I)
    if I in _memo:
        return _memo[I]
    orb = clean_orbit(I)
    # Survivor = _target_key-MAXIMAL member. `_target_key` uses |abs| and so TIES
    # integrals differing only in sign pattern (e.g. [1,0,-1,..] vs [-1,0,1,..]); those
    # are reduction-equivalent, so the choice among them only needs to be DETERMINISTIC
    # (else canonical_rep is non-idempotent). Append the signed tuple to break ties.
    # Corners have distinct |abs| (no ties), so this never changes the corner result.
    rep = I if orb is None else max(orb, key=lambda j: (tkey(j), j))
    _memo[I] = rep
    return rep
