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
