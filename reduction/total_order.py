"""THE total order. One definition, imported everywhere -- never re-implemented.

WHY THIS MODULE EXISTS. The same key was written out by hand in a dozen places:
six `_target_key` (greedy_reduce, beam_search_v7/v8/v9/truthcull,
sailir_symmetry), seven `tkey` (symmetry_route, canonical_rep, gate_engine2 and
four build/verify scripts), each with its own copy of the sector-mask loop. They
drifted: canonical_rep.tkey builds the mask with `range(8)`, so on gravity3L --
which has TEN denominators -- it disagreed with the live order on 13,729 of
20,000 random integrals. A reduction cache mixes entries from the workers and
from symmetry routing, and it is acyclic ONLY because both descend in the SAME
order; two orders in one cache is a cycle waiting to happen.

THE ORDER. tkey(i) = (-sector_rank, -r, -s, |abs|-tuple)
  SMALLER tkey == HIGHER in the ordering == what gets eliminated first.
  A valid reduction sends an integral to terms with strictly LARGER tkey.

Note this deliberately runs opposite to ibp_env.weight(), which is larger =
higher with |abs| already negated. Do NOT derive one from the other: that needs
a per-component sign fix, and getting it wrong is the exact defect ORDERING.md
records (45/55 rise-fall vs 98/2 correct).

SAILIR_SECTOR_RANK=1 selects the adopted sector-senior order; unset is the
legacy order. It must be identical for orchestrator, workers, symmetry_route and
data-gen within a run.
"""
import os

from topo_config import N_DEN as _N_DEN
from topo_config import N_IND as _N_IND
from topo_config import TOPO_DIR as _TOPO_DIR

_SECTOR_RANK = os.environ.get('SAILIR_SECTOR_RANK', '0') == '1'
if _SECTOR_RANK:
    from sector_rank import RANK_IDX as _RANK_IDX


def sector_mask(i):
    """Bitmask of which denominators are present. Width comes from the topology
    -- hardcoding 8 here is what made canonical_rep disagree on gravity3L."""
    m = 0
    for k in range(_N_DEN):
        if i[k] > 0:
            m |= 1 << k
    return m


def base_key(i):
    """The order WITHOUT the sector-rank prefix: (-r, -s, |abs|-tuple).

    This is not merely "tkey with the flag off" -- it is what the scripts that
    BUILD the sector ranking must use, since ranking sectors with a key that
    depends on the sector ranking is circular. Kept here so those scripts import
    it instead of hand-copying the tuple, which is how the copies drifted.
    """
    return (-sum(x for x in i if x > 0), -sum(-x for x in i if x < 0),
            tuple(abs(x) for x in i))


def tkey(i):
    """The workers' total order. Smaller = higher = eliminated first."""
    base = base_key(i)
    if not _SECTOR_RANK:
        return base
    return (-_RANK_IDX[sector_mask(i)],) + base


# Historical alias: the workers call it _target_key, the router calls it tkey.
_target_key = tkey


# ---------------------------------------------------------------------------
# ZERO SECTORS. A sector whose propagators supply no scale is identically zero
# in dimensional regularisation. Which sectors those are is not a propagator
# count -- it is a property of the topology, and Kira already computed it:
# <TOPO_DIR>/kira_std_all/sectormappings/<family>/trivialsector.
#
# VALIDATED against 6,000 completed workers before being trusted, with ZERO
# counterexamples:
#   Kira ZERO    -> worker returned nothing (59), or only zero-sector terms
#                   (211/211, all 1,266 terms) -- i.e. 0 = 0, valid but useless
#   Kira NONZERO -> worker never returned an empty expression (0 of 5,730)
#
# For gravity3L, 784 of the 1,024 sectors spanned by the 10 real denominators
# are zero (76.6%): ALL of L0-L3, 207/210 of L4, 223/252 of L5. Measured cost of
# not knowing this: 10.4% of workers targeted a zero integral, and 19.4% of all
# children produced land in a zero sector -- each becoming a future job that
# spawns more.
#
# Stored as a flat bool array indexed by the 15-bit mask: 32 KB, O(1) lookup,
# no hashing. This is called per term over a 1.78M-term expression.
_ZERO_SECTORS = None


def _load_zero_sectors():
    """Kira's trivialsector list as a bool array, or None if unavailable."""
    import numpy as _np
    base = os.path.join(_TOPO_DIR, 'kira_std_all', 'sectormappings')
    try:
        fams = [d for d in os.listdir(base)
                if os.path.isfile(os.path.join(base, d, 'trivialsector'))]
    except OSError:
        return None
    if not fams:
        return None
    path = os.path.join(base, fams[0], 'trivialsector')
    arr = _np.zeros(1 << _N_IND, dtype=bool)
    n = 0
    try:
        for tok in open(path).read().replace('\n', ',').split(','):
            tok = tok.strip()
            if tok:
                v = int(tok)
                if 0 <= v < arr.size:
                    arr[v] = True
                    n += 1
    except Exception:
        return None
    print(f'[zero-sectors] {n:,} zero sectors from {path}', flush=True)
    return arr


def is_zero_integral(i):
    """True if `i` lies in a sector that vanishes identically.

    Falls back to the unconditional no-propagator test when Kira's list is not
    available for this topology: with no propagator at all the integrand is a
    polynomial in the loop momenta and there is no scale to carry the dimension.
    """
    global _ZERO_SECTORS
    if _ZERO_SECTORS is None:
        _ZERO_SECTORS = _load_zero_sectors()
        if _ZERO_SECTORS is None:
            _ZERO_SECTORS = False          # sentinel: use the fallback
    if _ZERO_SECTORS is False:
        for k in range(_N_DEN):
            if i[k] > 0:
                return False
        return True
    m = 0
    for k in range(_N_IND):
        if i[k] > 0:
            m |= 1 << k
    return bool(_ZERO_SECTORS[m])
