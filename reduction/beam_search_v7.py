#!/usr/bin/env python
"""v6 beam search: macro-beam architecture on top of v5 strip-passenger.

Built from v5 (`beam_search_v5.py`). v5 wastes beam capacity because:
  - The model only sees expr + RS_KEYS (not RS values; see prepare_batched
    _input_v5_dummy). When 40 beam states share the same expr_fp and share
    ~249/250 RS keys (as empirical inspection of probe_84_v5_lazyrs at step
    250 confirmed), the model returns essentially the same scores for all
    40 — they all pick the same top-K action.
  - Tabu adds a tiny bit of artificial variation but the beam still collapses
    to 1-3 distinct exprs / 40 slots.

v6 changes:
  1. At the top of each step, group beam by `expr_fp = frozenset(expr.items())`
     and keep ONE representative per macro (the highest-score variant).
  2. Run the model ONCE per (macro, target) pair — not per (state, target).
  3. Expand top-K actions per macro into candidates; per-expr cap during
     beam selection enforces real expr diversity in the next beam.

What we keep from v5:
  - strip-passenger semantics (apply_substitution_v5, add_sub_to_resolved_v5)
  - aux_flat / LAZY_RS / iraws-keep-first / tabu
  - --beam-sort {weight,mixed}
  - any() termination on first state with nm=0
"""
import argparse
import math
import os
import pickle
import select
import sys
import time
from collections import namedtuple


def _cap_incidental_threads():
    """When running parallel (>1 enumerate workers OR >1 model threads), pin the
    incidental numpy/OpenMP/BLAS threadpools to 1 thread and park idle threads.

    WHY: those pools default to the node's FULL core count (e.g. 128) and idle
    OpenMP/BLAS threads spin-wait (OMP_WAIT_POLICY=ACTIVE) by default. Each forked
    enumerate worker is a separate process with its own pool, and torch's pool in
    the main spins during the fork phase -> total CPU drifts past the requested
    cores and Condor HOLDS the job ("cpu usage exceeded RequestCpus"). The model
    keeps its threads via torch.set_num_threads(--n-threads), which is independent
    of these caps (verified: t_step unchanged with caps on).

    MUST run before numpy/torch import (pools size themselves at import/first use),
    so we peek argv here rather than wait for argparse. setdefault() respects any
    value the caller set explicitly. Only fires for the parallel configs (the
    single-worker/single-thread path is untouched)."""
    av = sys.argv

    def _val(flag):
        for i, a in enumerate(av):
            if a == flag and i + 1 < len(av):
                return av[i + 1]
            if a.startswith(flag + '='):
                return a.split('=', 1)[1]
        return None
    try:
        nw = int(_val('--n-workers') or 1)
        nt = int(_val('--n-threads') or 1)
    except ValueError:
        nw = nt = 1
    ncap = max(1, nw, nt)                       # == request_cpus
    if nw > 1 or nt > 1:
        for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ.setdefault(k, '1')
        os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
        # Route MKL through GNU OpenMP (libgomp) instead of Intel OpenMP
        # (libiomp5). Intel's runtime throws a fatal assertion
        # (kmp_affinity.cpp) when a forked child's available procs disagree with
        # what it detected at init — which is exactly our case (fork pool +
        # sched_setaffinity). libgomp is the same runtime torch already uses,
        # respects sched_setaffinity + OMP_NUM_THREADS, and (verified) keeps the
        # model 8-threaded. Do NOT use MKL_THREADING_LAYER=SEQUENTIAL: it forces
        # the model matmuls to 1 thread. Do NOT set KMP_AFFINITY=disabled.
        os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
    # HARD CAP via CPU affinity (env thread-caps alone do NOT keep total CPU <=
    # request: numpy ops in the maxweight metric + each forked worker inherently
    # use slightly >1 core, so usage drifts ~1.2x past request -> Condor holds.
    # Verified: pinning the tree to N cores makes the OS confine ALL threads /
    # forked children to N cores -> measured CPU CANNOT exceed N). Fire ALWAYS
    # (a 1-cpu worker's numpy metric wants ~1.2 cores too). Pin to ncap cores at
    # a PID-derived offset so MANY workers sharing a node spread across cores
    # instead of all piling on core 0. If Condor already cpuset-isolates the job
    # (len(allowed)==ncap) we skip and let its cpuset stand. SAILIR_NO_CPU_PIN=1.
    if os.environ.get('SAILIR_NO_CPU_PIN', '0') != '1':
        _msg = ''
        try:
            allowed = sorted(os.sched_getaffinity(0))
            n = len(allowed)
            if 0 < ncap < n:
                start = (os.getpid() * ncap) % n
                want = {allowed[(start + i) % n] for i in range(ncap)}
                os.sched_setaffinity(0, want)
                got = sorted(os.sched_getaffinity(0))
                _msg = (f"PINNED ncap={ncap} allowed={n} -> got {got}"
                        if set(got) == want else
                        f"PIN-MISMATCH ncap={ncap} want={sorted(want)} got={got}")
            else:
                _msg = (f"NO-PIN ncap={ncap} allowed={n} "
                        f"(allowed<=ncap; relying on cgroup cpuset)")
        except Exception as _e:   # noqa: BLE001 - want to SEE any failure
            _msg = f"PIN-FAILED ncap={ncap}: {type(_e).__name__}: {_e}"
        sys.stderr.write(f"[CPU-PIN] {_msg}\n")
        sys.stderr.flush()


_cap_incidental_threads()

import numpy as np
import torch

# Resolve paths relative to this file (reduction/), not hardcoded, so the repo
# can live anywhere: reduction/ -> repo root for `sailir`, plus this dir for siblings.
_BS7_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_BS7_DIR))   # repo root (SAILIR_phase2) for `sailir`
sys.path.insert(0, _BS7_DIR)                     # this dir (reduction/) for siblings

from sailir import ibp_env
from sailir.topology import Topology
from sailir.ibp_env import (
    set_prime, set_paper_masters_only, init_from_topology, IBPEnvironment,
    apply_resolved_subs, apply_substitution_target_only,
    enumerate_valid_actions_with_indirect_cache, compute_indirect_substituted,
    compute_indirect_substituted_with_aux,
    weight, is_master, solve_ibp_for,
)


def _aux_to_result(aux_flat, env):
    """Reconstruct compute_indirect_substituted result list from aux tuple.

    iraws entries are 3-tuples (sub_int, op, shift). Raws are looked up
    fresh from env._raw_eq_cache (recomputed on miss) — only the cache
    persistently holds raw equations, so eviction can free memory.
    """
    cached_unique, union_bms, raw_id_to_idx, indirect_raws = aux_flat
    if not indirect_raws:
        return []
    n_indices = len(indirect_raws[0][0])
    result = []
    for sub_int, ibp_op, shift in indirect_raws:
        seed = tuple(sub_int[i] - shift[i] for i in range(n_indices))
        raw = env.get_raw_equation_cached(ibp_op, seed)
        idx = raw_id_to_idx[(ibp_op, seed)]
        # v7 Stage 2b: cu entries stay PackedEq in the result — the packed
        # enumerate consumes them directly (id-membership searchsorted), so NO
        # dict conversion here (that per-step transient is what capped the
        # memory win and cost the time).
        result.append((sub_int, ibp_op, shift, raw,
                       cached_unique[idx], union_bms[idx]))
    return result


_PICKLABLE_AUX_MARKER = '__v5_aux_v2__'


def _aux_to_picklable(aux_flat):
    """rid is (op, seed)-keyed; iraws are 3-tuples (sub_int, op, shift).
    All pickle-stable. Append marker for format detection on load.
    """
    if aux_flat is None:
        return None
    cu, ubm, rid, iraws = aux_flat
    # v7: PackedEq cu -> dict cu so the checkpoint is registry-independent
    # (re-interned on load via _aux_from_picklable).
    if cu and isinstance(cu[0], PackedEq):
        cu = [c.to_dict(_V7_REGISTRY) for c in cu]
    return (cu, ubm, rid, iraws, _PICKLABLE_AUX_MARKER)


def _aux_from_picklable(aux_flat, env=None):
    """rid stays (op, seed)-keyed; iraws are 3-tuples (no raw stored)."""
    if aux_flat is None:
        return None
    if len(aux_flat) == 5 and aux_flat[4] == _PICKLABLE_AUX_MARKER:
        cu, ubm, rid, iraws, _ = aux_flat
        # v7: re-intern dict cu -> PackedEq
        if cu and isinstance(cu[0], dict):
            cu = [PackedEq.from_dict(c, _V7_REGISTRY) for c in cu]
        return (cu, ubm, rid, iraws)
    # Legacy 4-tuple format — assume same-process; just pass through.
    return aux_flat


def _prune_aux_by_recency(aux_flat, recent_sub_ints):
    """Drop iraws entries whose sub_int (anchor) isn't in recent_sub_ints.
    Rebuild cu/ubm/rid from the kept raws.

    Args:
        aux_flat: (cached_unique, union_bms, raw_key_to_idx, indirect_raws)
            where raw_key_to_idx is keyed by (op, seed) tuples and
            indirect_raws entries are 3-tuples (sub_int, op, shift).
        recent_sub_ints: iterable of sub_int tuples to KEEP.

    Returns the pruned aux tuple (unchanged tuple if nothing dropped).
    """
    cu, ubm, rid, iraws = aux_flat
    if not iraws:
        return aux_flat
    keep_set = set(recent_sub_ints)
    new_iraws = [e for e in iraws if e[0] in keep_set]
    if len(new_iraws) == len(iraws):
        return aux_flat  # nothing to drop
    # Rebuild cu/ubm/rid: keep only entries referenced by surviving iraws.
    # Key is (op, seed) where seed = sub_int - shift componentwise.
    referenced = set()
    for sub_int, op, shift in new_iraws:
        n = len(sub_int)
        seed = tuple(sub_int[i] - shift[i] for i in range(n))
        referenced.add((op, seed))
    new_cu = []
    new_ubm = []
    new_rid = {}
    for key, old_idx in rid.items():
        if key in referenced:
            new_rid[key] = len(new_cu)
            new_cu.append(cu[old_idx])
            new_ubm.append(ubm[old_idx])
    return (new_cu, new_ubm, new_rid, new_iraws)
# DO NOT `from ... import PRIME` — set_prime() rebinds the module attr but
# not local bindings. Always read ibp_env.PRIME so we pick up the configured value.
from sailir.classifier import IBPActionClassifier
from beam_search_utils import get_non_masters, get_sector_mask, filter_to_sector

# ── v7: packed-int/GF(p) cu representation ──────────────────────────────────
# cu entries are stored as PackedEq (int32 ids / int16 coeffs) in the aux —
# the persistent 40x-smaller form. Consumers (enumerate / solve) still see
# dicts: _aux_to_result converts the (pruned, one-state-at-a-time) cu to dicts
# transiently. One IntegralRegistry per worker process (module global).
from sailir.packed_eq import IntegralRegistry, PackedEq
from sailir import packed_cu as _packed_cu
from sailir.packed_rs_ops import (
    add_sub_to_resolved_packed, apply_resolved_subs_dict_x_packed)

_V7_REGISTRY = None          # IntegralRegistry, set in main()
_PACKED_RS = False           # Stage 3b: when True, State.resolved_subs values are
                             # PackedEq (int32/int16) instead of dicts. Set in
                             # main() from --packed-rs / SAILIR_PACKED_RS.
_V7_PACKED_RS_CACHE = None   # {sub_int: (ids, coeffs)}, grown lazily


def _v7_pack_cu(dict_cu):
    """Convert a list of {integral: coeff} dict cu entries -> list[PackedEq]."""
    return [PackedEq.from_dict(c, _V7_REGISTRY) for c in dict_cu]


def _v7_aux_pack(aux_flat):
    """Pack a dict-cu aux (from a dict compute_indirect) into packed-cu aux."""
    cu, ubm, rid, iraws = aux_flat
    return (_v7_pack_cu(cu), ubm, rid, iraws)


def _v7_as_dict(eq):
    """A single cu entry -> dict (convert if PackedEq) for solve_ibp_for."""
    return eq.to_dict(_V7_REGISTRY) if isinstance(eq, PackedEq) else eq


def _remap_packed_eq(pe, base_N, remap_arr):
    """Relabel a PackedEq's worker-local new ids (>= base_N) to global ids via
    remap_arr (indexed by id - base_N), re-sorting to keep the ascending-id
    invariant. ids < base_N (the shared registry base at fork) are untouched.

    No-op (returns pe unchanged) when the entry has no new id — since ids are
    sorted, that's just ids[-1] < base_N. Within one worker the new ids are
    distinct tuples -> distinct globals, and globals (>= base_N) never collide
    with retained base ids (< base_N), so the remap is a pure relabel+sort with
    no coefficient recombination."""
    ids = pe.ids
    if ids.shape[0] == 0 or ids[-1] < base_N:
        return pe
    new_ids = ids.astype(np.int32, copy=True)
    mask = ids >= base_N
    new_ids[mask] = remap_arr[ids[mask] - base_N]
    order = np.argsort(new_ids, kind='stable')
    return PackedEq(np.ascontiguousarray(new_ids[order]),
                    np.ascontiguousarray(pe.coeffs[order]))


def _remap_state(c, base_N, remap_arr):
    """Remap every PackedEq carrying a worker-local new id in a materialized
    survivor: the aux_flat cu (the registering part) and — defensively — the
    resolved_subs values (mat_rs is registry-neutral, so these are no-ops). ubm/
    rid/iraws are tuple/index-keyed (bitmask is tuple-derived), so they're
    portable as-is. cu order is preserved, so rid's cu-index map stays valid."""
    new_rs = {k: _remap_packed_eq(v, base_N, remap_arr)
              for k, v in c.resolved_subs.items()}
    new_c = c._replace(resolved_subs=new_rs)
    if c.aux_flat is not None:
        cu, ubm, rid, iraws = c.aux_flat
        new_cu = [_remap_packed_eq(pe, base_N, remap_arr) for pe in cu]
        new_c = new_c._replace(aux_flat=(new_cu, ubm, rid, iraws))
    return new_c


# ============================================================================
# v5 State
# ============================================================================

State_v5 = namedtuple(
    'State_v5',
    ['expr', 'resolved_subs',
     'score', 'path', 'n_non_masters',
     'max_w12', 'total_w12',  # cached sort keys, computed once at construction
     'aux_flat',  # (cached_unique, union_bms, raw_id_to_idx, indirect_raws)
                  # or None to mean "rebuild from scratch on use"
     'last_prob'],  # model prob of the action that CREATED this state, for
                    # beam_sort='prob'. None on states not made by an action
                    # (the root), which sort last.
)
State_v5.__new__.__defaults__ = (None, None, None, None)  # max_w12, total_w12, aux_flat, last_prob

# Weight of the n_non_masters regulariser in beam_sort='prob_nm'.
# cost = _NM_LAMBDA * nm - log(prob). See sort_prob_nm for the scale argument.
_NM_LAMBDA = float(os.environ.get('SAILIR_BEAM_NM_LAMBDA', '0.03'))


# ============================================================================
# Weight-active partition  —  SINGLE-STEP REDUCTION SEMANTICS  (READ THIS FIRST)
# ============================================================================
# !!! THIS BEAM SEARCH IS ALWAYS RUN AS A SINGLE-STEP (ONE-WEIGHT-LEVEL)       !!!
# !!! REDUCER. IT DOES *NOT* REDUCE THE START INTEGRAL ALL THE WAY TO MASTERS. !!!
#
# Every expression is STRIPPED to the "active bucket" (is_active, below) AS IT
# IS BUILT (see apply_substitution_v5), so get_non_masters() / n_non_masters
# count ONLY the active non-masters — NOT every non-master integral. Therefore
# `n_non_masters == 0` means "the ACTIVE BUCKET is drained", i.e. the start
# integral has been reduced by ONE weight level into a "passenger expansion": a
# combination of strictly-lower-weight integrals (the passengers, recovered by
# replay). The ORCHESTRATOR chains these one-step reductions level by level. A
# single worker NEVER has "reach the masters" as its goal.
#
# DO NOT read `n_non_masters == 0` as "reduced to masters." It is "active bucket
# drained = one level done." The ground-truth training generator
# (generate_multisector_data.py) DOES reduce fully to masters and has NO strip;
# that is a DIFFERENT loop — do not transplant its semantics onto this code.
# ============================================================================

# SAILIR_SECTOR_RANK=1 -> the adopted sector-senior order (reduction/ORDERING.md):
# sector rank first, then (r,s), then |abs|. Under it the active bucket is
# same-sector only: every subsector term is a passenger regardless of its (r,s),
# because a proper subsector ranks strictly below its parent. _START_SECTOR is the
# start integral's sector mask, set by the worker entrypoint before the search
# (None -> legacy behavior even with the flag on, e.g. library users).
# Must be set identically for orchestrator AND workers of a run (shared order).
_SECTOR_RANK = os.environ.get('SAILIR_SECTOR_RANK', '0') == '1'
_START_SECTOR = None
if _SECTOR_RANK:
    from sector_rank import RANK_IDX as _RANK_IDX

# denominator count for _sector_mask (topology-keyed via SAILIR_TOPOLOGY,
# default pentagonbox = 8, the historical hardwired value). main() asserts it
# matches the loaded topology.
from topo_config import N_DEN as _TC_N_DEN

# ============================================================================
# SAILIR_SYM_DROP -- DO NOT ENABLE IN PRODUCTION. VERDICT LOCKED 2026-07-11.
# ============================================================================
# The same-(r,s) symmetry DROP DETECTOR: an active integral at the START's (r,s)
# level whose within-sector image eliminates it into strictly-lower terms is
# dropped from the active expression (passenger; the orchestrator router
# provably disposes of it from the replayed output -- no cycle possible).
#
# It was implemented, validated (masters identical on m1/m2/m3), AND MEASURED
# TO BE A NULL RESULT: the detector fired 352 times across the three A/B probes
# yet shortened NOTHING (m2 782 vs 782 steps -- literally identical paths;
# m1 -0.2%, m3 -0.8%), while ADDING worker CPU (+60% on m2's short workers,
# dominated by the per-worker symmetry-engine load). The dropped integrals sit
# off the critical path: the beam's chosen eliminations never depended on them.
# Population measurements agree: ~8% of active-expression integrals and ~0.9%
# of candidate-introduced integrals are below-threshold eliminable, and the
# winning-path witness rate is ~0 -- the orchestrator routing already extracts
# everything extractable without the model's cooperation.
#
# PRODUCTION POLICY (dshih): NO symmetrization / level-drop checking inside
# one-step workers. Symmetry lives at the ORCHESTRATOR only
# (canonical_monolithic_rule). Within-sector relations reach workers only as
# ACTIONS at the retrain, where the model can play them deliberately.
# This flag is kept solely for retrain-era experiments. See the note
# (symmetry_inference_routing.tex) and analysis/logs/cmp_symdrop_v1.log.
_SYM_DROP = os.environ.get('SAILIR_SYM_DROP', '0') == '1'
_START_RS = None
_START_INT = None
_drop_wt = None
_drop_memo = {}


def _init_sym_drop(start_int):
    """Set up the drop detector for this worker's target sector (call at startup,
    after ibp_env init). Loads the within-sector transform list once."""
    global _START_RS, _START_INT, _drop_wt
    w = weight(start_int)
    _START_RS = (w[0], w[1])
    _START_INT = tuple(start_int)
    from symmetry_route import _within_transforms
    _drop_wt = _within_transforms(_sector_mask(start_int))


def _has_drop_witness(integral):
    """True iff some within-sector map yields a single value identity that
    eliminates `integral` into strictly-lower-(r,s) terms: the image's same-level
    part is either empty (I = lower), or exactly {I: c} with c != 1 (odd reflection
    c = -1, or any (1-c)I = lower). Memoized."""
    out = _drop_memo.get(integral)
    if out is not None:
        return out
    from canonicalize import image_unsigned, P as _CP
    out = False
    for (M, c) in _drop_wt:
        img = image_unsigned(integral, M, c)
        if img is None:
            continue
        good = True
        self_co = 0
        for J, co in img.items():
            wj = weight(J)
            if (wj[0], wj[1]) == _START_RS:
                if J != integral:
                    good = False
                    break
                self_co = co % _CP
        if good and self_co != 1:
            out = True
            break
    _drop_memo[integral] = out
    return out


def _sector_mask(i):
    m = 0
    for k in range(_TC_N_DEN):
        if i[k] > 0:
            m |= 1 << k
    return m


def is_active(integral, start_w12):
    """True iff integral is in the ACTIVE BUCKET for the single-step reduction
    (see the banner above). Legacy: weight (w1, w2) lex-≥ start_w12. Sector-senior
    (SAILIR_SECTOR_RANK=1 with _START_SECTOR set): additionally same sector as the
    start — subsector terms are passengers at any (w1, w2). With SAILIR_SYM_DROP=1
    (and _init_sym_drop called): integrals AT the start's (r,s) carrying the
    symmetry drop witness are passengers too — the orchestrator router eliminates
    them for free from the replayed output."""
    if _SECTOR_RANK and _START_SECTOR is not None \
            and _sector_mask(integral) != _START_SECTOR:
        return False
    w = weight(integral)
    if (w[0], w[1]) < start_w12:
        return False
    if _SYM_DROP and _drop_wt is not None and (w[0], w[1]) == _START_RS \
            and integral != _START_INT and _has_drop_witness(integral):
        return False
    return True


def strip_passenger(d, start_w12):
    """Return new dict containing only active-weight entries (drops passengers
    = sub-weight integrals, which the one-step reducer leaves for the next
    level / the replay)."""
    return {k: v for k, v in d.items() if is_active(k, start_w12)}


# --- Single-step SUCCESS criterion -----------------------------------------
# Default success = the (w1,w2) active bucket is drained (n_non_masters == 0).
# SAILIR_SUCCESS_TOTAL=1 switches the success test to the FULL TOTAL ORDERING:
# the start is reduced as soon as NO non-master remains at-or-above the start in
# the total ordering (-r, -s, |abs|). The total-active set is a SUBSET of the
# (w1,w2)-active set, so this fires no later than the default -> the single step
# is even shorter (strictly <= the steps of the default criterion).
_SUCCESS_TOTAL = os.environ.get('SAILIR_SUCCESS_TOTAL', '0') == '1'
_START_TOTAL_KEY = None   # set in main() = _target_key(start_int)


def _target_key(i):
    """Total-ordering key matching the training-data target selection:
    (-r, -s, |abs|-tuple). Smaller = higher in the total ordering.

    DESIGN DECISION 2026-07-10 (see reduction/ORDERING.md): the adopted order is
    SECTOR RANK first, then (r,s), then |abs|. Default = LEGACY order;
    SAILIR_SECTOR_RANK=1 switches to the adopted order (rank prefix). The flag
    must be set identically for orchestrator + workers + symmetry_route +
    canonical_rep of a run (the shared order keeps the substitution cache
    acyclic)."""
    w = weight(i)
    if not _SECTOR_RANK:
        return (-w[0], -w[1], w[2])
    return (-_RANK_IDX[_sector_mask(i)], -w[0], -w[1], w[2])


def _is_success(s, target_sector):
    """Single-step success test. Default: (w1,w2) active bucket drained. With
    SAILIR_SUCCESS_TOTAL=1: no non-master remains at-or-above the start in the
    full total ordering ('reduce by total weight')."""
    if not _SUCCESS_TOTAL:
        return s.n_non_masters == 0
    if s.n_non_masters == 0:
        return True   # (w1,w2)-active drained => total-active (a subset) drained
    for k in get_non_masters(s.expr, target_sector):
        if _target_key(k) <= _START_TOTAL_KEY:
            return False
    return True


def _rs_as_dict(resolved_subs):
    """Dict view {sub_int: {integral: coeff}} of resolved_subs for the dict-only
    from-scratch compute path. Packed (Stage 3b) -> materialize each PackedEq
    value to a dict (transient, one state at a time); else pass through."""
    if _PACKED_RS and resolved_subs:
        reg = _V7_REGISTRY
        return {k: v.to_dict(reg) for k, v in resolved_subs.items()}
    return resolved_subs


# ============================================================================
# v5 substitution math (Option F + weight strip)
# ============================================================================

def apply_substitution_v5(expr_t, sub_int, sol, target_sector, start_w12):
    """Apply substitution, keeping ONLY active target-sector terms in expr:
      - active target-sector terms → new_expr_t
      - passenger target-sector terms (below start_w) → discarded
      - sub-sector terms → discarded
    All discarded (sub-weight + sub-sector) spillover is recovered by
    replay_full_expr at the end (the worker's final_expr), so nothing needs to
    be accumulated here. (Removed the old sub_accum / Option-F bucket: it was
    write-only and redundant with replay.)
    """
    if sub_int not in expr_t:
        return expr_t
    coeff = expr_t[sub_int]
    new_expr_t = {k: v for k, v in expr_t.items() if k != sub_int}
    nd = ibp_env.N_DENOMINATORS
    for integral, sub_coeff in sol.items():
        new_coeff = (coeff * sub_coeff) % ibp_env.PRIME
        if new_coeff == 0:
            continue
        # Sub-sector spillover -> discarded (recovered by replay)
        in_target_sector = True
        for i in range(nd):
            if (1 if integral[i] > 0 else 0) != target_sector[i]:
                in_target_sector = False
                break
        if not in_target_sector:
            continue
        # Target-sector passenger (lower weight) -> discarded (recovered by replay)
        if not is_active(integral, start_w12):
            continue
        if integral in new_expr_t:
            s = (new_expr_t[integral] + new_coeff) % ibp_env.PRIME
            if s == 0:
                del new_expr_t[integral]
            else:
                new_expr_t[integral] = s
        else:
            new_expr_t[integral] = new_coeff
    return new_expr_t


def add_sub_to_resolved_v5(resolved_subs, target, sol, start_w12):
    """Like add_sub_to_resolved but keeps every value dict stripped of
    passenger-weight entries. `sol` may be unstripped; we strip after every
    write.
    """
    if _PACKED_RS:
        # Stage 3b: resolved_subs values are PackedEq. sol arrives as a dict
        # (from solve_ibp_for); pack it and delegate to the verified packed op.
        reg = _V7_REGISTRY
        sol_packed = sol if hasattr(sol, 'ids') else PackedEq.from_dict(sol, reg)
        active_id = lambda i: is_active(reg.get_tuple(i), start_w12)  # noqa: E731
        return add_sub_to_resolved_packed(
            resolved_subs, target, sol_packed, reg, active_id, ibp_env.PRIME)
    # Step 1: resolve sol against existing RS, then strip passenger
    resolved_sol = apply_resolved_subs(sol, resolved_subs)
    resolved_sol = strip_passenger(resolved_sol, start_w12)

    # Step 2: shallow-copy outer dict, add new entry
    new_resolved = dict(resolved_subs)
    new_resolved[target] = resolved_sol

    # Step 3: propagate target rewrite into existing entries that contain it
    for key, old_value in resolved_subs.items():
        if target not in old_value:
            continue
        value = dict(old_value)
        coeff = value.pop(target)
        for k, v in resolved_sol.items():
            new_coeff = (coeff * v) % ibp_env.PRIME
            if k in value:
                s = (value[k] + new_coeff) % ibp_env.PRIME
                if s == 0:
                    del value[k]
                else:
                    value[k] = s
            elif new_coeff != 0:
                value[k] = new_coeff
        # Strip again (resolved_sol is already stripped but safety)
        new_resolved[key] = strip_passenger(value, start_w12)
    return new_resolved


def apply_action_v5(state, target, ibp_op, delta, action_prob,
                    env, target_sector, start_w12,
                    use_incremental_aux=True,
                    lazy_rs=True):
    """Apply one (target, ibp_op, delta) action to a v5 State.
    Returns a (new_state, sol) tuple, or None on failure.

    sol is the raw IBP sol (active+passenger) that the action produces.
    Caller stores sol alongside (parent, target) so survivors can have
    their resolved_subs materialized post-selection.

    With lazy_rs=True (default), child.resolved_subs is left as None.
    add_sub_to_resolved_v5 is the expensive per-RS-value propagation step.
    Materialize for survivors only via `_materialize_lazy_rs()` after
    beam selection.
    """
    n_idx = ibp_env.N_INDICES
    seed = tuple(target[i] + delta[i] for i in range(n_idx))
    # v7 Stage 2b: REUSE the already-resolved cu entry when present (indirect
    # actions). compute_indirect's invariant (target_sector=None, entry not
    # pruned): cu[(ibp_op,seed)] == apply_resolved_subs(raw, resolved_subs), so
    # this is bit-identical and skips the re-resolution that grows with
    # |resolved_subs|. Falls back to recompute on a cu miss (direct/pruned).
    cached = None
    if state.aux_flat is not None:
        _idx = state.aux_flat[2].get((ibp_op, seed))
        if _idx is not None:
            cached = _v7_as_dict(state.aux_flat[0][_idx])
    if cached is None:
        raw = env.get_raw_equation_cached(ibp_op, seed)
        cached = (apply_resolved_subs_dict_x_packed(
                      raw, state.resolved_subs, _V7_REGISTRY, ibp_env.PRIME)
                  if _PACKED_RS else apply_resolved_subs(raw, state.resolved_subs))
    if target not in cached or cached[target] == 0:
        return None
    sol = solve_ibp_for(cached, target)
    if sol is None:
        return None
    new_expr = apply_substitution_v5(
        state.expr, target, sol, target_sector, start_w12,
    )

    if lazy_rs:
        new_rs = None  # to be materialized post-selection for survivors only
    else:
        new_rs = add_sub_to_resolved_v5(
            state.resolved_subs, target, sol, start_w12,
        )

    new_path = state.path + [(target, ibp_op, delta)]
    new_score = state.score + math.log(action_prob + 1e-10)
    nm = get_non_masters(new_expr, target_sector)
    mw, tw = _mw_tw_from_nm(nm)   # beam-ranking weights (total ordering if SAILIR_BEAM_TOTAL)
    child = State_v5(
        expr=new_expr,
        resolved_subs=new_rs,
        score=new_score,
        path=new_path,
        n_non_masters=len(nm),
        max_w12=mw,
        total_w12=tw,
        aux_flat=None,
        last_prob=action_prob,
    )
    return child, sol


def _materialize_lazy_rs(child, parent, target, sol, start_w12):
    """Materialize child.resolved_subs via add_sub_to_resolved_v5(parent.RS, target, sol).
    Lossless — same inputs/outputs as if apply_action_v5 had run with lazy_rs=False.
    """
    if child.resolved_subs is not None:
        return child  # already materialized
    new_rs = add_sub_to_resolved_v5(
        parent.resolved_subs, target, sol, start_w12,
    )
    return child._replace(resolved_subs=new_rs)


def _attach_incremental_aux(child, parent, target, env, target_sector,
                            use_exprkeyed=True, iraws_window=None,
                            iraws_keep_first=None):
    """Compute child's aux_flat from parent's aux_flat + the action target.
    Returns a new State_v5 with aux_flat set. Skips if parent's aux is None.

    use_exprkeyed: if True (default), use the bounded-anchor exprkeyed delta
        (iraws bounded by |useful K| ≈ |expr_nm|, not by |RS|). Otherwise use
        the depth-keyed compute_indirect_substituted_incremental.
    """
    if parent.aux_flat is None:
        return child  # can't do incremental; will rebuild from scratch on use
    new_resolved_sol = child.resolved_subs[target]
    # v7: always the PACKED depth-keyed path (exprkeyed is a dead end — it
    # explodes per-step time at depth). parent.aux_flat carries PackedEq cu.
    _result, new_aux = _packed_cu.compute_indirect_substituted_incremental_packed(
        parent.aux_flat, target, new_resolved_sol, child.resolved_subs,
        env.ibp_t, env.li_t, env.shifts, env._raw_eq_cache,
        _V7_REGISTRY, ibp_env.PRIME, ibp_env.N_INDICES,
        ibp_env, ibp_env.get_raw_equation, target_sector=None)
    # Optional: prune iraws to first N sub_ints (anchor-as-bootstrap) AND/OR
    # last N sub_ints (recency).
    if iraws_keep_first is not None or iraws_window is not None:
        keys = list(child.resolved_subs.keys())
        keep = set()
        if iraws_keep_first is not None:
            keep.update(keys[:iraws_keep_first])
        if iraws_window is not None:
            keep.update(keys[-iraws_window:])
        if len(keep) < len(keys):
            new_aux = _prune_aux_by_recency(new_aux, keep)
    return child._replace(aux_flat=new_aux)


# ============================================================================
# Model input with dummy subs (v5 variant)
# ============================================================================

def prepare_batched_input_v5_dummy(batch_data, device, max_subs=50,
                                    max_actions=900):
    """Batch model input.
    batch_data: list of (expr, resolved_subs, valid_actions, target_sector, target)

    Differences vs prepare_batched_input_v5:
    - No `subs` arg; dummy subs are RS keys with empty replacement dicts
      (variant F from v5_test_dummy_subs.py — verified |logit Δ|=1.5e-5,
      |prob Δ|=4.1e-7, top-1 always matches)
    - max_actions defaults to 900 (matches training distribution)
    """
    # MAX_REPLACEMENT_TERMS was previously imported from beam_search_full;
    # inlined here to keep v6 self-contained (beam_search_full now in archive/).
    MAX_REPLACEMENT_TERMS = 20  # Must match training

    batch_size = len(batch_data)
    max_terms = max(len(filter_to_sector(d[0], d[3])) for d in batch_data) if batch_data else 1
    max_terms = max(max_terms, 1)
    max_actions_eff = max(len(d[2]) for d in batch_data)
    max_actions_eff = min(max_actions_eff, max_actions)

    N = ibp_env.N_INDICES
    D = ibp_env.N_DENOMINATORS

    expr_integrals_np = np.zeros((batch_size, max_terms, N), dtype=np.int64)
    expr_coeffs_np = np.zeros((batch_size, max_terms), dtype=np.int64)
    expr_mask_np = np.zeros((batch_size, max_terms), dtype=bool)
    sub_keys_np = np.zeros((batch_size, max_subs, N), dtype=np.int64)
    sub_repl_ints_np = np.zeros((batch_size, max_subs, MAX_REPLACEMENT_TERMS, N), dtype=np.int64)
    sub_repl_coeffs_np = np.zeros((batch_size, max_subs, MAX_REPLACEMENT_TERMS), dtype=np.int64)
    sub_repl_mask_np = np.zeros((batch_size, max_subs, MAX_REPLACEMENT_TERMS), dtype=bool)
    sub_mask_np = np.zeros((batch_size, max_subs), dtype=bool)
    action_ibp_ops_np = np.zeros((batch_size, max_actions_eff), dtype=np.int64)
    action_deltas_np = np.zeros((batch_size, max_actions_eff, N), dtype=np.int64)
    action_mask_np = np.zeros((batch_size, max_actions_eff), dtype=bool)
    sector_masks_np = np.zeros((batch_size, D), dtype=bool)
    target_integrals_np = np.zeros((batch_size, N), dtype=np.int64)

    for i, (expr, resolved_subs, valid_actions, target_sector, target) in enumerate(batch_data):
        sector_expr = filter_to_sector(expr, target_sector)
        expr_items = list(sector_expr.items())
        n_expr = len(expr_items)
        if n_expr > 0:
            expr_integrals_np[i, :n_expr] = [integral for integral, _ in expr_items]
            expr_coeffs_np[i, :n_expr] = [coeff for _, coeff in expr_items]
            expr_mask_np[i, :n_expr] = True

        # Dummy subs: RS keys (last 50), empty replacement → all-zero repl tensors
        rs_items = list(resolved_subs.keys())[-max_subs:]
        for j, key_integral in enumerate(rs_items):
            sub_mask_np[i, j] = True
            sub_keys_np[i, j] = key_integral
            # leave sub_repl_ints/coeffs/mask as zeros (verified equivalent)

        n_actions = min(len(valid_actions), max_actions_eff)
        if n_actions > 0:
            action_ibp_ops_np[i, :n_actions] = [op for op, _ in valid_actions[:n_actions]]
            action_deltas_np[i, :n_actions, :] = [delta for _, delta in valid_actions[:n_actions]]
            action_mask_np[i, :n_actions] = True

        sector_masks_np[i] = get_sector_mask(target)
        target_integrals_np[i] = target

    return {
        'expr_integrals': torch.from_numpy(expr_integrals_np).to(device),
        'expr_coeffs': torch.from_numpy(expr_coeffs_np).to(device),
        'expr_mask': torch.from_numpy(expr_mask_np).to(device),
        'sub_keys': torch.from_numpy(sub_keys_np).to(device),
        'sub_repl_ints': torch.from_numpy(sub_repl_ints_np).to(device),
        'sub_repl_coeffs': torch.from_numpy(sub_repl_coeffs_np).to(device),
        'sub_repl_mask': torch.from_numpy(sub_repl_mask_np).to(device),
        'sub_mask': torch.from_numpy(sub_mask_np).to(device),
        'action_ibp_ops': torch.from_numpy(action_ibp_ops_np).to(device),
        'action_deltas': torch.from_numpy(action_deltas_np).to(device),
        'action_mask': torch.from_numpy(action_mask_np).to(device),
        'sector_mask': torch.from_numpy(sector_masks_np).to(device),
        'target_integral': torch.from_numpy(target_integrals_np).to(device),
    }


# ============================================================================
# Beam search loop
# ============================================================================

# SAILIR_BEAM_TOTAL=1: rank the beam by the FULL total ordering instead of just
# (w1,w2). max_w12 -> (r, s, -|abs|) of the heaviest non-master (larger=heavier),
# total_w12 -> (Sum r, Sum s, -Sum|abs|). Default (unset) = v7 (w1,w2) ranking.
# Also flips the maxweight action-clip metric to total weight (see _select_actions).
_BEAM_TOTAL = os.environ.get('SAILIR_BEAM_TOTAL', '0') == '1'


def _mw_tw_from_nm(nm):
    """(max_w12, total_w12) over a non-master iterable. Single source of truth so
    construction / resume / initial / the helpers can't drift. Default = (w1,w2);
    SAILIR_BEAM_TOTAL=1 = full total ordering (larger=heavier)."""
    if not nm:
        return ((0, 0, ()), (0, 0, ())) if _BEAM_TOTAL else ((0, 0), (0, 0))
    if not _BEAM_TOTAL:
        wl = [(weight(k)[0], weight(k)[1]) for k in nm]
        return max(wl), (sum(w[0] for w in wl), sum(w[1] for w in wl))
    sr = ss = 0
    asum = None
    best_tk = None
    best_w = None
    for k in nm:
        w = weight(k)
        r = w[0]
        s = w[1]
        a = w[2]
        sr += r
        ss += s
        if asum is None:
            asum = list(a)
        else:
            for j in range(len(a)):
                asum[j] += a[j]
        tk = (-r, -s, a)
        if best_tk is None or tk < best_tk:
            best_tk = tk
            best_w = w
    mw = (best_w[0], best_w[1], tuple(-x for x in best_w[2]))
    tw = (sr, ss, tuple(-x for x in asum))
    return mw, tw


def max_w12(expr, target_sector):
    mw, _ = _mw_tw_from_nm(get_non_masters(expr, target_sector))
    return mw


def total_w12(expr, target_sector):
    _, tw = _mw_tw_from_nm(get_non_masters(expr, target_sector))
    return tw


# ===========================================================================
# Action-cap selection strategies (experiment; env SAILIR_ACTION_SELECT)
# ---------------------------------------------------------------------------
# When a (parent,target) task has > max_actions valid actions, only max_actions
# are fed to the model. Default 'first900' = valid[:max_actions] = the OLDEST
# actions (enumerate order is Phase-1a directs then Phase-1b indirects in iraws
# append/oldest->newest order); left as-is so prepare_batched_input still caps it
# (bit-identical). The other strategies pre-select here, post-tabu:
#   last900    newest actions (valid[-max_actions:])
#   maxweight  lowest max (w1,w2) over the action's RHS  (#2)
#   shortest   fewest RHS terms                          (#3)
#   sumweight  lowest (Sum w1, Sum w2) over the RHS       (#4)
# Metric is over the resolved relation's RHS AFTER modding out subweight (already
# stripped: cu built with min_w12 raw-strip), subsector and masters and the
# target itself — i.e. exactly get_non_masters(relation, target_sector), matching
# the single-step max_w12/total_w12 criterion.
#
# EXPERIMENT FINDINGS (pentagonbox 74/84/longrunner/memhog, 8/8): wall-time, not
# step count, is the real metric. first900 wins on easy (74,84) and the
# first900-friendly hard case (longrunner). The metric strategies are a MAJOR win
# (~14-16x faster) exactly when first900 is pathological (memhog: 720 steps @
# 13.7s/step -> ~170 steps @ 3.6s/step). Among the metrics, MAXWEIGHT is the
# robust all-rounder / consensus winner: lowest SUMMED wall-time across the suite
# (~1.9x faster than first900) because it captures the memhog win without
# shortest's catastrophic longrunner blowup (10205s). last900 (newest) is bad even
# with directs kept. Default stays first900 (bit-identical); set maxweight via env
# for the pathological/compute-dominating integrals. The metric compute itself is
# free (74: identical 4.3s/step); cost differences are which states the path
# visits. NOTE: known train/inference target-ordering mismatch — training data
# (generate_multisector_data.py) targets ONE integral by full (r,s,lex) order;
# this beam search targets ALL (r,s)-tied integrals. Alignment is the next step.
_ACTION_SELECT = os.environ.get('SAILIR_ACTION_SELECT', 'first900')
_METRIC_STRATEGIES = ('maxweight', 'shortest', 'sumweight')
# per-id metric arrays, cached + grown like packed_cu._bm_array (bounded memory)
_METRIC_CACHE = {'n': -1, 'twk': None, 'wt1': None, 'wt2': None,
                 'ism': None, 'sec': None}


def _secbm_of(integral):
    """Sector bitmask matching beam_search_utils.get_sector_mask (skip ISP)."""
    bm = 0
    j = 0
    isp = set(ibp_env.ISP_POSITIONS)
    for i in range(ibp_env.N_INDICES):
        if i in isp:
            continue
        if integral[i] > 0:
            bm |= (1 << j)
        j += 1
    return bm


def _metric_arrays(reg):
    """Per-id (wt1, wt2, ism, sec) int arrays for the metric strategies. Append-
    only registry -> rebuild only when it grows (one bounded set of arrays)."""
    n = len(reg._to_tuple)
    if _METRIC_CACHE['n'] == n:
        return _METRIC_CACHE
    old = _METRIC_CACHE['n'] if _METRIC_CACHE['n'] > 0 else 0
    twk = np.empty(n, np.int64)   # packed total-weight scalar (larger = heavier)
    wt1 = np.empty(n, np.int64)
    wt2 = np.empty(n, np.int64)
    ism = np.empty(n, np.uint8)
    sec = np.empty(n, np.int64)
    if old:
        twk[:old] = _METRIC_CACHE['twk']
        wt1[:old] = _METRIC_CACHE['wt1']
        wt2[:old] = _METRIC_CACHE['wt2']
        ism[:old] = _METRIC_CACHE['ism']
        sec[:old] = _METRIC_CACHE['sec']
    ABS44 = (1 << 44) - 1
    for i in range(old, n):
        t = reg._to_tuple[i]
        w = weight(t)
        wt1[i] = w[0]
        wt2[i] = w[1]
        ap = 0                              # packed |abs|, index 0 most-significant
        for x in t:
            ax = abs(x)
            ap = (ap << 4) | (ax if ax < 15 else 15)
        twk[i] = (int(w[0]) << 50) | (int(w[1]) << 44) | (ABS44 - ap)
        ism[i] = 1 if is_master(t) else 0
        sec[i] = _secbm_of(t)
    _METRIC_CACHE.update(n=n, twk=twk, wt1=wt1, wt2=wt2, ism=ism, sec=sec)
    return _METRIC_CACHE


def _select_actions(valid, target, aux_flat, max_actions, strategy):
    """Pick which <=max_actions actions reach the model. valid is a list of
    (op, delta); aux_flat = (cu, ubm, rid, iraws) of the parent. Bit-identical
    no-op for 'first900' (handled by the caller, never invoked)."""
    nv = len(valid)
    if nv <= max_actions:
        return valid
    cu = aux_flat[0]
    rid = aux_flat[2]
    N = ibp_env.N_INDICES
    if strategy == 'last900':
        # Keep ALL directs (not in rid, same as the metric strategies) + the LAST
        # (max_actions - n_directs) indirects (newest in enumerate order), in
        # original order. Fixes the old valid[-max_actions:], which sliced off the
        # front where the directs live -> doubly broken (no directs + tail only).
        direct_idx, indirect_idx = [], []
        for ai in range(nv):
            op, delta = valid[ai]
            seed = tuple(target[i] + delta[i] for i in range(N))
            (direct_idx if rid.get((op, seed)) is None
             else indirect_idx).append(ai)
        nkeep = max_actions - len(direct_idx)
        sel = (sorted(direct_idx + indirect_idx[-nkeep:]) if nkeep > 0
               else sorted(direct_idx[:max_actions]))
        return [valid[i] for i in sel]
    # metric strategies: recover each action's RHS via rid[(op, target+delta)].
    arr = _metric_arrays(_V7_REGISTRY)
    twk, wt1, wt2, ism, sec = (arr['twk'], arr['wt1'], arr['wt2'],
                               arr['ism'], arr['sec'])
    tid = _V7_REGISTRY.get_id(target)
    tsec = int(sec[tid])
    BIGW = 1 << 20
    keys = np.empty(nv, dtype=np.int64)
    for ai in range(nv):
        op, delta = valid[ai]
        seed = tuple(target[i] + delta[i] for i in range(N))
        idx = rid.get((op, seed))
        if idx is None:
            keys[ai] = -(1 << 62)          # direct / not in cu -> always keep
            continue
        ids = cu[idx].ids
        # in-target-sector non-master RHS terms (exclude target itself)
        m = (sec[ids] == tsec) & (ism[ids] == 0) & (ids != tid)
        sel_ids = ids[m]
        if sel_ids.shape[0] == 0:
            keys[ai] = -(1 << 61)          # pure-master/sub RHS -> reduces -> keep
            continue
        if strategy == 'maxweight':
            if _BEAM_TOTAL:
                keys[ai] = int(twk[sel_ids].max())   # highest TOTAL-weight RHS term
            else:
                w1 = wt1[sel_ids]
                j = int(np.lexsort((wt2[sel_ids], w1))[-1])
                keys[ai] = int(w1[j]) * BIGW + int(wt2[sel_ids][j])
        elif strategy == 'shortest':
            keys[ai] = int(sel_ids.shape[0])
        else:  # sumweight
            keys[ai] = int(wt1[sel_ids].sum()) * BIGW + int(wt2[sel_ids].sum())
    sel = np.argpartition(keys, max_actions)[:max_actions]
    sel.sort()                              # keep original order among the chosen
    return [valid[i] for i in sel]


def beam_search_v5(env, model, start_expr, target_sector, start_w12,
                   beam_width=40, max_steps=1000, device='cpu',
                   beam_sort='mixed', max_actions=900, ckpt_path=None,
                   ckpt_every=50, verbose=True,
                   resume_from=None, tabu=False,
                   use_incremental_aux=True,
                   use_exprkeyed=True,
                   ckpt_every_step=False,
                   iraws_window=None,
                   iraws_keep_first=None,
                   lazy_rs=True,
                   model_batch_chunk=8,
                   top_k=None,
                   n_workers=1):
    """Sequential beam search, v5 strip-active semantics.

    n_workers: if >1, parallelize ONLY the per-step enumerate phase (P1) across
               this many forked processes (the GIL-bound enumerate loop can't be
               thread-parallelized). At n_workers==1 the original serial enumerate
               path runs verbatim — the fork-pool is never constructed, so the
               optimized single-worker code path is byte-for-byte unchanged and
               carries ZERO pooling overhead. Complements --n-threads (which
               parallelizes the model_fwd phase via torch threads); the two use
               the same cores in disjoint phases (enumerate vs model).

    resume_from: path to a ckpt.pkl (or result.pkl) from a prior run; resumes
                 the beam from there at its recorded step.
    tabu: if True, maintain per-expr action tabu: never re-pick the same
          (target, ibp_op, delta) from the same expr fingerprint. Matches
          delta_beam_search.py's DELTA_TABU semantics.
    """
    resume_step = 0
    if resume_from is not None and os.path.exists(resume_from):
        with open(resume_from, 'rb') as f:
            d = pickle.load(f)
            if isinstance(d, dict) and d.get('_streamed'):
                # streamed checkpoint: meta dict, then n_states state-dicts as
                # independent pickle frames (see _stream_dump_ckpt).
                _n = d['n_states']
                d = dict(d)
                d['beam'] = [pickle.load(f) for _ in range(_n)]
        resume_step = d['step']
        beam = []
        # Convert saved aux from picklable (op, seed)-keyed form back to id()-
        # keyed in-memory form. This preserves the EXACT iraws structure the
        # original run had (including incremental's (op, seed) dedup). The
        # alternative of force-rebuilding emits extra Phase-1b actions under
        # multiple anchors for the same (op, seed) raw, diverging from the
        # original trajectory.
        for s in d['beam']:
            if s.get('aux_flat') is not None:
                s['aux_flat'] = _aux_from_picklable(s['aux_flat'], env=env)
            # Recompute cached weight fields if missing
            if 'max_w12' not in s or 'total_w12' not in s:
                _nm = get_non_masters(s['expr'], target_sector)
                s['max_w12'], s['total_w12'] = _mw_tw_from_nm(_nm)
            s.pop('sub_accum', None)   # tolerate pre-removal checkpoints
            beam.append(State_v5(**s))
        if verbose:
            print(f'[v6 RESUME] loaded {len(beam)} beam states from step '
                  f'{resume_step}', flush=True)
    else:
        initial_nm = get_non_masters(start_expr, target_sector)
        init_mw, init_tw = _mw_tw_from_nm(initial_nm)
        initial = State_v5(
            expr=dict(start_expr),
            resolved_subs={},
            score=0.0,
            path=[],
            n_non_masters=len(initial_nm),
            max_w12=init_mw,
            total_w12=init_tw,
        )
        beam = [initial]
    best_state = beam[0]
    initial_mw = max_w12(start_expr, target_sector)
    t0 = time.time()

    # Tabu: expr fingerprint -> set of (target, ibp_op, delta) tried before
    tabu_dict = {} if tabu else None
    # Restore tabu from ckpt if present (resume bit-identical needs this).
    if (resume_from is not None and os.path.exists(resume_from)
            and tabu_dict is not None and d.get('tabu_dict')):
        saved_tabu = d['tabu_dict']
        # saved as list of (expr_items, [(target, op, delta), ...])
        for expr_items, entries in saved_tabu:
            tabu_dict[frozenset(expr_items)] = {
                (tuple(t), op, tuple(dlt)) for t, op, dlt in entries
            }
        if verbose:
            n_entries = sum(len(v) for v in tabu_dict.values())
            print(f'[v6 RESUME] restored tabu_dict: {len(tabu_dict)} expr '
                  f'fingerprints, {n_entries} total entries', flush=True)
    # Free the loaded checkpoint dict — beam + tabu have been extracted into
    # live State_v5 objects, so keeping `d` bound pins the ENTIRE serialized
    # beam (GBs of aux) alive for the whole run (diagnostic artifact + leak).
    if resume_from is not None and os.path.exists(resume_from):
        del d

    def sort_weight(s):
        return (s.max_w12, s.n_non_masters, -s.score)

    def sort_totalweight(s):
        return (s.total_w12, s.n_non_masters, -s.score)

    def sort_prob_nm(s):
        # 'prob_nm': the model's current-step probability GUIDES, with an
        # n_non_masters REGULARISER -- a single cost, not a lexicographic key:
        #
        #     cost = lambda * nm  -  log(prob)          (lower is better)
        #
        # Still no accumulated path score and no max_w12, so this isolates the
        # effect of the nm term alone against plain 'prob'.
        #
        # SCALE, measured on the bce/sigmoid checkpoint over 200 states: the top
        # action of every state scores ~0.9999 (whole range 0.967-1.000), so
        # -log(prob) spans only ~0.033, while nm spans 1..18. Any lambda much
        # above ~0.002 therefore makes nm strictly dominant and the sort
        # degenerates to lexicographic-by-nm. The default below is chosen so one
        # extra non-master costs about as much as the FULL observed spread of
        # top-action confidence -- i.e. an exceptionally confident action can
        # just offset one extra non-master, and no more. Tune with
        # SAILIR_BEAM_NM_LAMBDA.
        p = s.last_prob if s.last_prob is not None else 1e-12
        return _NM_LAMBDA * s.n_non_masters - math.log(max(p, 1e-12))

    def sort_prob(s):
        # beam_sort='prob': rank ONLY by the model's probability for the action
        # that produced this state. Deliberately drops both of the other terms
        # in sort_weight: no max_w12/n_non_masters progress regulariser, and no
        # accumulated path score -- so nothing carries over between steps and
        # the frontier is whatever looks most confident right now.
        return (-(s.last_prob if s.last_prob is not None else -1.0),)

    # Hoist memprobe config out of the per-step loop. Per-step we want at
    # most a set-membership check or an integer modulo — no env-var parsing.
    _memprobe_path = os.environ.get('SAILIR_MEMPROBE_FULL')
    _memprobe_steps_set = None
    _memprobe_every = None
    _memprobe_start = None
    if _memprobe_path:
        _steps_env = os.environ.get('SAILIR_MEMPROBE_FULL_STEPS')
        if _steps_env:
            _memprobe_steps_set = {int(x) for x in _steps_env.split(',') if x.strip()}
        else:
            _memprobe_every = int(os.environ.get(
                'SAILIR_MEMPROBE_FULL_EVERY', '10'))
            _memprobe_start = int(os.environ.get(
                'SAILIR_MEMPROBE_FULL_START', '20'))

    # Glibc memory tuning.
    #  (1) End-of-step malloc_trim(0) — DEFAULT ON. Returns the freed heap top
    #      to the OS; helped at shallow depth and is neutral at worst (it only
    #      ever reclaims the trimmable top). Opt out: SAILIR_END_OF_STEP_TRIM=0.
    #  (2) mmap-threshold pin via mallopt(M_MMAP_THRESHOLD=64MB) — DEFAULT OFF
    #      (OPT-IN: SAILIR_MMAP_THRESHOLD=<bytes>). It was in the validated
    #      probes but those only ran NORMAL integrals (74/84/LR); on the
    #      huge-equation HOG regime it BACKFIRES (+~3GB): forcing medium
    #      equation dicts onto the heap where live data blocks trim. So it is
    #      no longer applied unless explicitly requested.
    _end_of_step_trim = None
    try:
        import ctypes as _ctypes
        import ctypes.util as _ctypes_util
        _libc = _ctypes.CDLL(_ctypes_util.find_library('c'))
        _mmap_thr = os.environ.get('SAILIR_MMAP_THRESHOLD')
        if _mmap_thr:                                  # opt-in only
            _libc.mallopt.argtypes = [_ctypes.c_int, _ctypes.c_int]
            _libc.mallopt.restype = _ctypes.c_int
            _libc.mallopt(-3, int(_mmap_thr))          # M_MMAP_THRESHOLD = -3
        if os.environ.get('SAILIR_END_OF_STEP_TRIM', '1') != '0':
            _libc.malloc_trim.argtypes = [_ctypes.c_size_t]
            _libc.malloc_trim.restype = _ctypes.c_int
            _end_of_step_trim = _libc.malloc_trim
    except Exception:
        _end_of_step_trim = None

    # Optional beam-set trace: log per step (step, n_distinct_expr_fps,
    # hash_of_expr_fp_set) to verify the hypothesis "tabu trap fires only
    # when current beam's expr_fp set equals some prior step's set".
    # Gated by SAILIR_TRACE_BEAM_SETS=<path>. Per-step cost = one hash
    # over n_beam frozensets; per-step memory = 3 ints.
    _beam_set_trace_path = os.environ.get('SAILIR_TRACE_BEAM_SETS')
    _beam_set_history = []  # list of (step, n_distinct, set_hash)

    # Optional bounded-tabu mode (SAILIR_TABU_CAP=N, N>0). Per (expr_fp,target)
    # we accumulate every tried (op,delta) into a tabu set, exactly like the
    # baseline aggressive tabu, and pick top-K of valid minus that set. The ONE
    # difference: the effective block is capped at CAP actions. When the number
    # of currently-valid tabu'd actions exceeds CAP, we re-allow the highest-
    # model-prob overflow (keeping the lowest-prob CAP blocked) — so once we've
    # swept CAP-deep, the model's current favourites re-open and get re-tried
    # under the evolved RS (different sol → different children). Reduces to
    # baseline byte-for-byte until |valid∩tabu| first exceeds CAP. Replaces the
    # plain tabu filter — they're alternatives, not stackable. Storage grows
    # like baseline (the cap governs the FILTER, not the stored set).
    # Aggressive tabu is the FOUNDATION whenever --tabu is set: every top-K
    # pick is recorded and blocked (== baseline). SAILIR_TABU_CAP only adds the
    # optional cap+cycling LAYER on top: cap>0 caps the effective block at CAP
    # and re-allows the highest-prob overflow once swept CAP-deep; cap<=0 means
    # NO cap (effective block = ∞), i.e. pure aggressive baseline tabu.
    _tabu_cap = int(os.environ.get('SAILIR_TABU_CAP', '0'))
    _tabu_cap_eff = _tabu_cap if _tabu_cap > 0 else float('inf')
    _bounded_tabu_on = bool(tabu)
    _bt_tabu = {} if _bounded_tabu_on else None

    # Focused runtime memory breakdown (SAILIR_MEM_BREAKDOWN=1). Each sampled
    # step we RESET VmHWM at step start (/proc/self/clear_refs) so the print's
    # peak_rss is THIS step's transient peak (not the all-time max), then report:
    #   peak_rss   = per-step transient peak (pre end-of-step trim)
    #   rss        = post-trim quiescent RSS
    #   smaps      = where RSS lives: heap (brk) / anon (mmap) / file
    #   named structs (shared-seen partition): beam,cand,cand_meta,cache,bt,tabu
    #   glibc_transient = peak_rss - rss  (freed-then-trimmed each step)
    #   live_unaccounted = rss - sum_struct  (live, unnamed -> next target)
    # Every 4th sample a gc deep-walk-by-type names any live_unaccounted.
    _mem_bd = os.environ.get('SAILIR_MEM_BREAKDOWN')
    _mem_bd_every = int(os.environ.get('SAILIR_MEM_BREAKDOWN_EVERY', '25'))

    def _membd_deepsize(o, seen=None):
        import sys as _s
        from collections import deque as _dq
        if seen is None:
            seen = set()
        tot = 0; st = _dq([o])
        while st:
            x = st.popleft(); i = id(x)
            if i in seen:
                continue
            seen.add(i)
            if isinstance(x, np.ndarray):
                tot += x.nbytes; continue
            tot += _s.getsizeof(x, 0)
            if isinstance(x, dict):
                for k, v in x.items():
                    st.append(k); st.append(v)
            elif isinstance(x, (list, tuple, set, frozenset, _dq)):
                for e in x:
                    st.append(e)
            else:
                # arbitrary object: traverse its attributes so env/model/caches
                # held in __dict__ or __slots__ are actually counted.
                _xd = getattr(x, '__dict__', None)
                if _xd is not None:
                    st.append(_xd)
                _sl = getattr(type(x), '__slots__', None)
                if _sl:
                    for _n in _sl:
                        try:
                            st.append(getattr(x, _n))
                        except Exception:
                            pass
        return tot

    def _membd_mem():
        rss = hwm = 0
        with open('/proc/self/status') as _f:
            for _l in _f:
                if _l.startswith('VmRSS:'):
                    rss = int(_l.split()[1])
                elif _l.startswith('VmHWM:'):
                    hwm = int(_l.split()[1])
        return rss, hwm   # kB; hwm here is the per-step peak after a reset

    def _membd_reset_peak():
        # Reset VmHWM to current VmRSS so the next VmHWM read is THIS step's
        # peak. CLEAR_REFS_MM_HIWATER_RSS = '5'. Cheap; no side effects beyond
        # the peak counter (which only the diagnostic relies on).
        try:
            with open('/proc/self/clear_refs', 'w') as _cf:
                _cf.write('5')
        except Exception:
            pass

    def _membd_smaps():
        # Sum Rss (kB) by region category from /proc/self/smaps:
        #   heap = the main-arena brk region ([heap]); anon = anonymous mmaps
        #   (glibc secondary arenas + large malloc/numpy/torch); file = mapped
        #   files (libs, model). Tells WHERE the RSS physically lives.
        heap = anon = filed = 0
        is_heap = is_file = False
        try:
            with open('/proc/self/smaps') as _f:
                for _l in _f:
                    # Mapping header lines start with "start-end" (a '-' in the
                    # first token); detail lines are "Field:  N kB".
                    if '-' in _l.split(' ', 1)[0]:
                        _p = _l.split()
                        _path = _p[5] if len(_p) >= 6 else ''
                        is_heap = (_path == '[heap]')
                        is_file = _path.startswith('/')
                    elif _l.startswith('Rss:'):
                        _kb = int(_l.split()[1])
                        if is_heap:
                            heap += _kb
                        elif is_file:
                            filed += _kb
                        else:
                            anon += _kb
        except Exception:
            pass
        return heap, anon, filed

    def _membd_torch_bytes():
        # Sum unique LIVE torch tensor storage (model params + forward
        # transients: probs_all, batch_data, ...). These live in anon-mmap,
        # OUTSIDE the Python heap, so _membd_deepsize can't see them — they are
        # the dominant PEAK driver on heavy candidate-gen steps.
        try:
            import torch as _t
            import gc as _g
            _seen_st = set()
            _tot = 0
            for _o in _g.get_objects():
                if isinstance(_o, _t.Tensor):
                    try:
                        _st = _o.untyped_storage()
                        _sid = id(_st)
                        if _sid not in _seen_st:
                            _seen_st.add(_sid)
                            _tot += _st.nbytes()
                    except Exception:
                        pass
            return _tot
        except Exception:
            return 0

    def _membd_mallinfo():
        # glibc allocator stats via malloc_info() XML (mallinfo2 doesn't exist
        # on glibc 2.28; old mallinfo() int fields overflow past 2GB). Reports
        # glibc-managed memory ONLY (large allocs / numpy / torch mmap; NOT
        # CPython pymalloc small-object arenas — those are counted by the gc
        # heap walk instead). free = glibc freed-but-retained (fragmentation);
        # mmap = bytes in mmap'd regions; system = total from-OS in glibc arenas.
        try:
            import ctypes as _c
            import re as _re
            _libc2 = _c.CDLL('libc.so.6')
            _buf = _c.c_char_p()
            _sz = _c.c_size_t()
            _libc2.open_memstream.restype = _c.c_void_p
            _libc2.open_memstream.argtypes = [
                _c.POINTER(_c.c_char_p), _c.POINTER(_c.c_size_t)]
            _fp = _libc2.open_memstream(_c.byref(_buf), _c.byref(_sz))
            _libc2.malloc_info.argtypes = [_c.c_int, _c.c_void_p]
            _libc2.malloc_info(0, _fp)
            _libc2.fflush(_c.c_void_p(_fp))
            _libc2.fclose(_c.c_void_p(_fp))
            _xml = _c.string_at(_buf.value, _sz.value).decode()
            _libc2.free.argtypes = [_c.c_void_p]
            _libc2.free(_c.cast(_buf, _c.c_void_p))

            def _last(_pat):
                _mm = _re.findall(_pat, _xml)
                return int(_mm[-1]) if _mm else 0
            _fast = _last(r'<total type="fast" count="\d+" size="(\d+)"')
            _rest = _last(r'<total type="rest" count="\d+" size="(\d+)"')
            _mmap = _last(r'<total type="mmap" count="\d+" size="(\d+)"')
            _syscur = _last(r'<system type="current" size="(\d+)"')
            _free = _fast + _rest
            return {'system': _syscur, 'hblkhd': _mmap, 'fordblks': _free,
                    'uordblks': max(_syscur - _free, 0)}
        except Exception:
            return None

    def _membd_gc_by_type(top_n=15):
        # FULL live-heap walk. CPython UNTRACKS dicts/tuples/lists whose contents
        # are all atomic (int/str) — gc.get_objects() does NOT list them, and
        # SAILIR's equation dicts (seed→int-coeff) are exactly that. So we seed
        # from the GC roots and traverse gc.get_referents(), which DOES reach
        # untracked children. id-dedup; numpy/torch buffers special-cased. This
        # reaches every reachable live object → total = the real python RSS, and
        # the by-type rows name the dominant container. Returns
        # (total_bytes, [(typename, bytes), ...]).
        import gc as _g
        import sys as _s
        from collections import Counter as _Cn
        try:
            import numpy as _np
        except Exception:
            _np = None
        try:
            import torch as _tt
        except Exception:
            _tt = None
        _g.collect()
        _seen = set()
        _tot = 0
        _bt = _Cn()
        _st_seen = set()
        _stack = _g.get_objects()                 # tracked roots
        _get_ref = _g.get_referents
        # Exclude the walk's OWN structures (seen/stack/bt/st_seen) so we never
        # count the instrumentation as heap (the seen-set alone is GBs on a big
        # heap — that observer effect inflated the total above rss).
        _self_ids = {id(_seen), id(_bt), id(_st_seen), id(_stack)}
        while _stack:
            _o = _stack.pop()
            _i = id(_o)
            if _i in _seen:
                continue
            _seen.add(_i)
            if _i in _self_ids:
                continue
            if _np is not None and isinstance(_o, _np.ndarray):
                _sz = _o.nbytes
            elif _tt is not None and isinstance(_o, _tt.Tensor):
                try:
                    _sid = id(_o.untyped_storage())
                    if _sid in _st_seen:
                        _sz = _s.getsizeof(_o, 0)
                    else:
                        _st_seen.add(_sid)
                        _sz = _o.untyped_storage().nbytes()
                except Exception:
                    _sz = _s.getsizeof(_o, 0)
            elif (_tt is not None and 'Storage' in type(_o).__name__
                  and hasattr(_o, 'nbytes')):
                # Storage object reached directly. getsizeof(storage) returns the
                # BUFFER size, so without this branch torch storage is counted
                # twice (once via its Tensor above). Dedup by storage id across
                # both paths so each buffer is counted exactly once.
                _sid = id(_o)
                if _sid in _st_seen:
                    _sz = 0
                else:
                    _st_seen.add(_sid)
                    try:
                        _sz = _o.nbytes()
                    except Exception:
                        _sz = _s.getsizeof(_o, 0)
            else:
                _sz = _s.getsizeof(_o, 0)
            _tot += _sz
            _typ = type(_o)
            _modn = getattr(_typ, '__module__', '')
            if not isinstance(_modn, str):
                _modn = ''
            _nm = getattr(_typ, '__name__', None)
            if not isinstance(_nm, str):
                _nm = repr(_typ)
            _bt[_modn + '.' + _nm] += _sz
            # Follow referents to reach UNTRACKED children (the equation dicts).
            for _r in _get_ref(_o):
                if id(_r) not in _seen:
                    _stack.append(_r)
        return _tot, _bt.most_common(top_n)

    # Diagnostic: dump per-task (expr_fp, target, n_v, picked_actions, V) every
    # step to <dir>/picks_step<N>.pkl. Used to diff baseline-tabu vs rank-cycle
    # picks and locate the first divergence. Off unless SAILIR_PICK_DUMP set.
    _pick_dump_dir = os.environ.get('SAILIR_PICK_DUMP')

    # Optional tabu audit / sol fallback infra.
    # SAILIR_TABU_AUDIT=1       → maintain _sol_history + dump audit on stuck
    # SAILIR_TABU_SOL_FALLBACK=1 → maintain _sol_history AND consult it on
    #                              stuck to let "drifted" actions through.
    # Either flag activates _sol_history. No cap on fallback uses — it only
    # fires when primary tabu has zero valid actions, so it's self-limiting.
    _tabu_audit_on = os.environ.get('SAILIR_TABU_AUDIT') == '1'
    _tabu_sol_fallback = os.environ.get('SAILIR_TABU_SOL_FALLBACK') == '1'
    # _sol_history[frozenset(expr_fp)][(target, op, delta)] = list of (step, sol_fp)
    _sol_history = ({} if (_tabu_audit_on or _tabu_sol_fallback) else None)
    _sol_fallback_count = 0

    # ----- enumerate fork-pool (ONLY constructed/entered when n_workers > 1) ---
    # The per-step enumerate phase (P1) is GIL-bound — dict membership, set/list
    # dedup, tuple building — so threads can't parallelize it; we fork processes
    # that inherit the beam copy-on-write and each enumerate a DISJOINT slice of
    # parents. The packed cu (numpy-backed PackedEq) keeps the beam to few Python
    # objects, so a child's read-only traversal dirties only small header pages,
    # not the big coefficient buffers — that is what makes fork-COW cheap here.
    # Bit-identical to the serial loop: same three filters, same seed/delta, same
    # per-parent target order; results are re-sorted by parent_idx (stable) so the
    # model batch sees the identical task order. aux_flat is a pure derived cache
    # of resolved_subs, so a worker that rebuilds it returns the packed aux for the
    # main process to store (perf memo) — correctness is independent of it.
    def _enum_slice(parent_indices, beam):
        local_tasks = []          # (parent_idx, target, valid) in per-parent order
        local_aux = {}            # parent_idx -> rebuilt PackedEq aux (rare path)
        for parent_idx in parent_indices:
            s = beam[parent_idx]
            if _is_success(s, target_sector):   # single-step: active bucket drained
                continue
            nm = get_non_masters(s.expr, target_sector)
            mw = max(((weight(k)[0], weight(k)[1])) for k in nm)
            tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
            dummy_subs = {k: {} for k in s.resolved_subs.keys()}
            if s.aux_flat is not None:
                indirect_cache = _aux_to_result(s.aux_flat, env)
            else:
                if (iraws_window is not None or iraws_keep_first is not None):
                    keys = list(s.resolved_subs.keys())
                    keep = set()
                    if iraws_keep_first is not None:
                        keep.update(keys[:iraws_keep_first])
                    if iraws_window is not None:
                        keep.update(keys[-iraws_window:])
                    if len(keep) < len(keys):
                        keep_list = [k for k in keys if k in keep]
                        fresh_dummy = {k: {} for k in keep_list}
                    else:
                        fresh_dummy = dummy_subs
                else:
                    fresh_dummy = dummy_subs
                indirect_cache, new_aux = compute_indirect_substituted_with_aux(
                    fresh_dummy, _rs_as_dict(s.resolved_subs),
                    env.ibp_t, env.li_t, env.shifts,
                    env._raw_eq_cache,
                )
                new_aux = _v7_aux_pack(new_aux)
                local_aux[parent_idx] = new_aux  # aux_flat not read below; just memo
            for target in tied:
                valid = _packed_cu.enumerate_valid_actions_with_indirect_cache_packed(
                    target, indirect_cache, s.resolved_subs,
                    env.ibp_t, env.li_t, env.shifts, 'subsector',
                    env._raw_eq_cache, _V7_REGISTRY, ibp_env.N_INDICES,
                )
                if (not _bounded_tabu_on
                        and tabu_dict is not None and valid):
                    expr_fp = frozenset(s.expr.items())
                    tabu_set = tabu_dict.get(expr_fp)
                    if tabu_set:
                        valid = [(op, dlt) for (op, dlt) in valid
                                 if (target, op, dlt) not in tabu_set]
                elif _bounded_tabu_on and valid:
                    _bt_key = (frozenset(s.expr.items()), target)
                    _bt_set = _bt_tabu.get(_bt_key)
                    if _bt_set:
                        _nblk = sum(1 for a in valid if a in _bt_set)
                        if _nblk < len(valid) and _nblk <= _tabu_cap_eff:
                            valid = [a for a in valid if a not in _bt_set]
                if valid:
                    if _ACTION_SELECT != 'first900':
                        valid = _select_actions(valid, target, s.aux_flat,
                                                max_actions, _ACTION_SELECT)
                    local_tasks.append((parent_idx, target, valid))
        return local_tasks, local_aux

    def _forkpool_map(slice_fn, partitions):
        # Generic fork-pool: one child per partition runs slice_fn(partition) and
        # pickles its result back through a pipe; the parent drains all pipes with
        # select() (avoids a pipe-full deadlock on large payloads), reaps, and
        # returns [result_0, result_1, ...] in partition order. Raises if any
        # worker fails. Children _exit (no atexit/buffer flush in the fork).
        # Shared by the enumerate (P1) and materialize (P4) parallel paths.
        children = []   # (pid, read_fd)
        for part in partitions:
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                # CHILD: inherits state COW; compute slice, pickle ('OK', result)
                # or ('ERR', traceback) to the pipe, _exit. Pickling is INSIDE the
                # try so an unpicklable result is reported, not silently lost.
                os.close(r)
                try:
                    payload = pickle.dumps(('OK', slice_fn(part)),
                                           pickle.HIGHEST_PROTOCOL)
                except BaseException:
                    import traceback
                    payload = pickle.dumps(('ERR', traceback.format_exc()),
                                           pickle.HIGHEST_PROTOCOL)
                try:
                    mv = memoryview(payload)
                    while mv:
                        nw = os.write(w, mv)
                        mv = mv[nw:]
                finally:
                    os.close(w)
                    os._exit(0)
            os.close(w)
            children.append((pid, r))
        bufs = {r: [] for _, r in children}
        open_fds = set(bufs)
        while open_fds:
            ready, _, _ = select.select(list(open_fds), [], [])
            for fd in ready:
                chunk = os.read(fd, 1 << 20)
                if chunk:
                    bufs[fd].append(chunk)
                else:
                    os.close(fd)
                    open_fds.discard(fd)
        results = []
        errs = []
        for pid, r in children:
            os.waitpid(pid, 0)
            raw = b''.join(bufs[r])
            if not raw:
                errs.append('worker produced no output')
                results.append(None)
                continue
            tag, val = pickle.loads(raw)
            if tag == 'ERR':
                errs.append(val)
                results.append(None)
            else:
                results.append(val)
        if errs:
            raise RuntimeError('fork-pool worker failed:\n'
                               + '\n---\n'.join(errs))
        return results

    def _round_robin_parts(n, n_workers):
        # Round-robin partition of range(n) across workers (rough cost balance);
        # callers that need original order re-sort, so the scheme is correctness-
        # neutral. Empty partitions (n < n_workers) are dropped.
        parts = [list(range(w, n, n_workers)) for w in range(n_workers)]
        return [p for p in parts if p]

    def _forkpool_enumerate(beam, n_workers):
        # Parallel P1: each worker enumerates a disjoint parent slice. Final task
        # order is restored by a stable sort on parent_idx (partition-independent).
        parts = _round_robin_parts(len(beam), n_workers)
        all_tasks = []
        aux_updates = {}
        for lt, la in _forkpool_map(lambda part: _enum_slice(part, beam), parts):
            all_tasks.extend(lt)
            aux_updates.update(la)
        all_tasks.sort(key=lambda t: t[0])  # stable -> per-parent target order kept
        return all_tasks, aux_updates

    for step in range(resume_step, max_steps):
        # Termination: ANY beam state has drained — we only need one successful
        # path (the proof). Don't wait for every beam slot to finish.
        winning = next((s for s in beam if _is_success(s, target_sector)), None)
        if winning is not None:
            best_state = winning
            if verbose:
                print(f'[v6 step {step}] beam state drained — DONE '
                      f'(path_len={len(winning.path)})', flush=True)
            break

        # Reset the peak-RSS counter at the START of a sampled step so the
        # MEMBD print reads THIS step's transient peak (the candidate-gen
        # spike), not the all-time high-water mark.
        if _mem_bd and (step + 1) % _mem_bd_every == 0:
            _membd_reset_peak()

        # Record per-step beam set for the trap-hypothesis verification.
        if _beam_set_trace_path is not None:
            _bs = frozenset(frozenset(s.expr.items()) for s in beam)
            _beam_set_history.append((step, len(_bs), hash(_bs)))

        # ── v6: macro-beam dedup ──────────────────────────────────────────
        # Collapse beam slots that share the same expr_fp. The model only
        # sees expr + RS_KEYS (not values; see prepare_batched_input_v5_dummy)
        # so duplicate-expr states return the same model scores → waste.
        # Keep best variant per expr_fp by progress key (low max_w12, low nm,
        # high score).
        _dedup_t = time.time()
        macro_groups = {}  # expr_fp → best State
        for s in beam:
            efp = frozenset(s.expr.items())
            cur = macro_groups.get(efp)
            if cur is None or (s.max_w12, s.n_non_masters, -s.score) < \
                              (cur.max_w12, cur.n_non_masters, -cur.score):
                macro_groups[efp] = s
        n_orig_beam = len(beam)
        beam = list(macro_groups.values())
        n_macros = len(beam)
        # Trace this collapse — it's the v6 telemetry the user cares about.
        if verbose and n_macros < n_orig_beam:
            print(f'[v6 step {step}] macro-dedup: '
                  f'{n_orig_beam} → {n_macros} distinct expr', flush=True)
        # ──────────────────────────────────────────────────────────────────

        # Build candidate list across all parents × valid actions
        candidates = []  # list of state_v5_child
        cand_metadata = []  # parallel list of (parent_state, target) for incremental aux
        all_state_summaries = []
        t_step = time.time()
        # V5_PROFILE=1: per-phase wall-clock instrumentation matching v4's
        # P1/P2/P3/P4 taxonomy from delta_beam_search.py.
        _v5_prof = bool(os.environ.get('V5_PROFILE'))
        # P1 = build tasks + valid actions: nm_tied + aux + gv + tabu
        # P2 = model batched scoring: batch_prep + model_fwd
        # P3 = apply children + beam selection: apply + sort
        # P4 = survivor materialize: attach_aux
        _p1 = {'nm_tied': 0.0, 'aux': 0.0, 'gv': 0.0, 'tabu': 0.0}
        _p2 = {'batch_prep': 0.0, 'model_fwd': 0.0}
        _p3 = {'apply': 0.0, 'sort': 0.0}
        _p4 = {'attach_aux': 0.0}
        _ct = {'aux_build_calls': 0, 'aux_repack_calls': 0,
               'enum_calls': 0, 'apply_calls': 0, 'attach_calls': 0,
               'n_iraws_total': 0, 'n_valid_total': 0,
               'cu_size_max': 0, 'rs_max': 0}

        # Per-state: compute indirect cache, enumerate tied targets, score actions
        # Build tasks for model batched inference
        if n_workers > 1:
            # Parallel enumerate: fork workers over disjoint parent slices.
            # The main process blocks in select() while workers run, so this
            # wall-clock IS the P1 phase duration (recorded under _p1['gv']).
            _t = time.time() if _v5_prof else 0
            tasks, _aux_updates = _forkpool_enumerate(beam, n_workers)
            for _pi, _na in _aux_updates.items():
                beam[_pi] = beam[_pi]._replace(aux_flat=_na)
            if _v5_prof:
                _p1['gv'] += time.time() - _t
        else:
            # === serial enumerate path — original code, indent-only change ===
            tasks = []  # (parent_idx, target, valid)
            for parent_idx, s in enumerate(beam):
                if _is_success(s, target_sector):   # single-step: active bucket drained
                    continue
                _t = time.time() if _v5_prof else 0
                nm = get_non_masters(s.expr, target_sector)
                mw = max(((weight(k)[0], weight(k)[1])) for k in nm)
                tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
                if _v5_prof:
                    _p1['nm_tied'] += time.time() - _t
                # Dummy subs (RS keys with empty values) — only used as keys for
                # enumerate_valid_actions iteration (it reads keys, not values).
                dummy_subs = {k: {} for k in s.resolved_subs.keys()}
                # Use cached aux_flat if available, else rebuild from scratch.
                _t = time.time() if _v5_prof else 0
                if s.aux_flat is not None:
                    indirect_cache = _aux_to_result(s.aux_flat, env)
                    if _v5_prof:
                        _ct['aux_repack_calls'] += 1
                else:
                    # v7: depth-keyed from-scratch rebuild (no exprkeyed), then pack
                    # the cu for storage. The returned indirect_cache stays dict
                    # (consumed transiently by enumerate); only new_aux is packed.
                    if (iraws_window is not None or iraws_keep_first is not None):
                        keys = list(s.resolved_subs.keys())
                        keep = set()
                        if iraws_keep_first is not None:
                            keep.update(keys[:iraws_keep_first])
                        if iraws_window is not None:
                            keep.update(keys[-iraws_window:])
                        if len(keep) < len(keys):
                            keep_list = [k for k in keys if k in keep]
                            fresh_dummy = {k: {} for k in keep_list}
                        else:
                            fresh_dummy = dummy_subs
                    else:
                        fresh_dummy = dummy_subs
                    indirect_cache, new_aux = compute_indirect_substituted_with_aux(
                        fresh_dummy, _rs_as_dict(s.resolved_subs),
                        env.ibp_t, env.li_t, env.shifts,
                        env._raw_eq_cache,
                    )
                    new_aux = _v7_aux_pack(new_aux)  # dict cu -> PackedEq for storage
                    if _v5_prof:
                        _ct['aux_build_calls'] += 1
                    beam[parent_idx] = s._replace(aux_flat=new_aux)
                    s = beam[parent_idx]
                if _v5_prof:
                    _p1['aux'] += time.time() - _t
                if _v5_prof:
                    _ct['n_iraws_total'] += len(indirect_cache)
                    if s.aux_flat:
                        _ct['cu_size_max'] = max(_ct['cu_size_max'],
                                                  len(s.aux_flat[0]))
                    _ct['rs_max'] = max(_ct['rs_max'], len(s.resolved_subs))
                for target in tied:
                    _t = time.time() if _v5_prof else 0
                    valid = _packed_cu.enumerate_valid_actions_with_indirect_cache_packed(
                        target, indirect_cache, s.resolved_subs,
                        env.ibp_t, env.li_t, env.shifts, 'subsector',
                        env._raw_eq_cache, _V7_REGISTRY, ibp_env.N_INDICES,
                    )
                    if _v5_prof:
                        _p1['gv'] += time.time() - _t
                        _ct['enum_calls'] += 1
                    # Tabu: filter out (target, op, delta) tried previously from
                    # this expr fingerprint.
                    if (not _bounded_tabu_on
                            and tabu_dict is not None and valid):
                        _t = time.time() if _v5_prof else 0
                        expr_fp = frozenset(s.expr.items())
                        tabu_set = tabu_dict.get(expr_fp)
                        if tabu_set:
                            valid = [(op, dlt) for (op, dlt) in valid
                                     if (target, op, dlt) not in tabu_set]
                        if _v5_prof:
                            _p1['tabu'] += time.time() - _t
                    # Aggressive tabu: remove this (expr_fp,target)'s tabu'd actions
                    # here — BEFORE the max_actions truncation — so the model sees
                    # the same tabu-free window as baseline (byte-identical). TWO
                    # cases keep the FULL list instead, deferring to the pick loop:
                    #  • EXHAUSTED (every valid action tabu'd): filtering would drop
                    #    the task (the trap → stuck). Keep the full list so the pick
                    #    loop CYCLES — re-picks the top-K best to retry under the
                    #    evolved RS instead of dying.
                    #  • OVER fixed cap (only when SAILIR_TABU_CAP>0): pick loop
                    #    re-allows the highest-prob overflow using model probs.
                    elif _bounded_tabu_on and valid:
                        _t = time.time() if _v5_prof else 0
                        _bt_key = (frozenset(s.expr.items()), target)
                        _bt_set = _bt_tabu.get(_bt_key)
                        if _bt_set:
                            _nblk = sum(1 for a in valid if a in _bt_set)
                            if _nblk < len(valid) and _nblk <= _tabu_cap_eff:
                                valid = [a for a in valid if a not in _bt_set]
                        if _v5_prof:
                            _p1['tabu'] += time.time() - _t
                    if valid:
                        if _ACTION_SELECT != 'first900':
                            valid = _select_actions(valid, target, s.aux_flat,
                                                    max_actions, _ACTION_SELECT)
                        if _v5_prof:
                            _ct['n_valid_total'] += len(valid)
                        tasks.append((parent_idx, target, valid))

        # Instrumentation: dump tasks/valid lists at a specific step
        _dump_step = os.environ.get('V5_DUMP_VALIDS_AT_STEP')
        if _dump_step and int(_dump_step) == step + 1:
            _dump_path = os.environ.get('V5_DUMP_PATH',
                                        f'/tmp/v5_valids_step{step+1}.pkl')
            with open(_dump_path, 'wb') as _df:
                pickle.dump([
                    (pi, tuple(tg), tuple((op, tuple(d)) for op, d in v))
                    for pi, tg, v in tasks
                ], _df)
            print(f'  [DUMP] valids at step {step+1} → {_dump_path}', flush=True)

        if not tasks:
            # SOL_FALLBACK: before declaring STUCK, retry filter using
            # sol_fp drift. For each raw candidate (op, dlt), compute the
            # current sol_fp under THIS state's RS. If it doesn't match
            # ANY historical sol_fp under (target, op, dlt) for this
            # expr_fp, that means the substitution we'd apply now is
            # different from what was tabu'd → allow.
            if _tabu_sol_fallback and _sol_history is not None:
                tasks = []
                n_unblocked_total = 0
                for parent_idx in range(len(beam)):
                    s = beam[parent_idx]
                    nm = get_non_masters(s.expr, target_sector)
                    if not nm:
                        continue
                    mw = max(((weight(k)[0], weight(k)[1])) for k in nm)
                    tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
                    expr_fp_ = frozenset(s.expr.items())
                    ts_ = (tabu_dict.get(expr_fp_)
                           if tabu_dict is not None else None)
                    hist_p = _sol_history.get(expr_fp_)
                    if hist_p is None:
                        continue
                    ic = (_aux_to_result(s.aux_flat, env)
                          if s.aux_flat is not None else None)
                    if ic is None:
                        continue
                    dummy_subs = {k: {} for k in s.resolved_subs.keys()}
                    cu_p = s.aux_flat[0]
                    rid_p = s.aux_flat[2]
                    N_ = ibp_env.N_INDICES
                    for tgt in tied:
                        raw_valid = _packed_cu.enumerate_valid_actions_with_indirect_cache_packed(
                            tgt, ic, s.resolved_subs,
                            env.ibp_t, env.li_t, env.shifts, 'subsector',
                            env._raw_eq_cache, _V7_REGISTRY, ibp_env.N_INDICES,
                        )
                        unblocked = []
                        for (op, dlt) in raw_valid:
                            # If primary tabu doesn't block it, keep as-is.
                            if ts_ is None or (tgt, op, dlt) not in ts_:
                                unblocked.append((op, dlt))
                                continue
                            # Primary blocks it — check sol_fp drift.
                            key_ = (tgt, op, dlt)
                            hist = hist_p.get(key_, [])
                            if not hist:
                                continue  # primary tabu'd via different path
                            seed_p = tuple(tgt[i] + dlt[i] for i in range(N_))
                            cu_idx_p = rid_p.get((op, seed_p))
                            if cu_idx_p is None:
                                continue
                            sol_p = solve_ibp_for(_v7_as_dict(cu_p[cu_idx_p]), tgt)
                            if sol_p is None:
                                continue
                            cur_sol_fp = hash(frozenset(sol_p.items()))
                            hist_sol_fps = {sfp for _s, sfp in hist}
                            if cur_sol_fp not in hist_sol_fps:
                                # Drifted: substitution differs from
                                # what was previously tabu'd → allow.
                                unblocked.append((op, dlt))
                        if unblocked:
                            tasks.append((parent_idx, tgt, unblocked))
                            n_unblocked_total += len(unblocked)
                if tasks:
                    _sol_fallback_count += 1
                    if verbose:
                        print(f'[v6 step {step}] TABU TRAP - sol_fp fallback '
                              f'#{_sol_fallback_count}: '
                              f'{n_unblocked_total} drifted actions unblocked',
                              flush=True)
                    # Fall through to model forward / action selection.
                else:
                    if verbose:
                        print(f'[v6 step {step}] no tasks — STUCK (sol_fp '
                              f'fallback found no drifted actions)', flush=True)
            if not tasks:
                if verbose:
                    print(f'[v6 step {step}] no tasks — STUCK', flush=True)
                # All the diagnostics below only fire if STILL stuck after
                # the sol_fp fallback above (or if fallback was disabled).
                _real_stuck = True
            else:
                _real_stuck = False
            # If beam-set tracing is on, identify prior steps whose beam
            # had the SAME set of expr_fps as the current (stuck) beam.
            if _real_stuck and _beam_set_trace_path is not None:
                _cur_bs = frozenset(frozenset(s.expr.items()) for s in beam)
                _cur_h = hash(_cur_bs)
                _prior_matches = [s for s, _n, h in _beam_set_history
                                  if h == _cur_h and s < step]
                with open(_beam_set_trace_path, 'w') as _tf:
                    _tf.write(f'stuck_step\t{step}\n')
                    _tf.write(f'stuck_n_distinct\t{len(_cur_bs)}\n')
                    _tf.write(f'prior_steps_with_same_set\t'
                               f'{",".join(map(str, _prior_matches))}\n')
                    _tf.write(f'# history (step, n_distinct, hash):\n')
                    for s, n, h in _beam_set_history:
                        _tf.write(f'{s}\t{n}\t{h}\n')
                print(f'  [BEAM_SET_TRACE] stuck-set seen earlier at steps: '
                      f'{_prior_matches} → '
                      f'{_beam_set_trace_path}', flush=True)
            # Comprehensive forensic dump of the stuck state. Gated by env
            # var SAILIR_DUMP_STUCK=<path> so it only fires when requested.
            _dump_stuck_path = (os.environ.get('SAILIR_DUMP_STUCK')
                                 if _real_stuck else None)
            if _dump_stuck_path:
                try:
                    _stuck_payload = {
                        'step': step,
                        'target_sector': target_sector,
                        'start_w12': start_w12,
                        'beam_width': beam_width,
                        'max_actions': max_actions,
                        # Tabu — every (target, op, delta) recorded
                        'tabu_dict': {
                            tuple(expr_fp): list(entries)
                            for expr_fp, entries in tabu_dict.items()
                        } if tabu_dict is not None else None,
                        'tabu_summary': ({
                            'n_expr_fps': len(tabu_dict),
                            'total_entries': sum(len(v) for v in tabu_dict.values()),
                        } if tabu_dict is not None else None),
                        # Full beam: every state's expr, RS, sub_accum, path,
                        # score, max_w12, total_w12, n_non_masters, aux_flat
                        'beam': [
                            {
                                'expr': dict(st.expr),
                                'resolved_subs': (dict(st.resolved_subs)
                                                   if st.resolved_subs else None),
                                'path': list(st.path) if st.path else [],
                                'score': float(st.score),
                                'max_w12': tuple(st.max_w12),
                                'total_w12': tuple(st.total_w12),
                                'n_non_masters': int(st.n_non_masters),
                                'aux_flat': (_aux_to_picklable(st.aux_flat)
                                              if st.aux_flat is not None else None),
                            }
                            for st in beam
                        ],
                        # Per-state, per-target: what enumerate_valid_actions
                        # gave us (raw) and what tabu LEFT (post-filter), so
                        # we can see EXACTLY how many candidates each target
                        # had vs how many tabu took out.
                        'per_state_per_target': [],
                        # Best state tracker
                        'best_state': ({
                            'path_len': int(best_state.path_len if hasattr(best_state, 'path_len') else len(best_state.path)),
                            'n_non_masters': int(best_state.n_non_masters),
                            'max_w12': tuple(best_state.max_w12),
                            'score': float(best_state.score),
                        } if best_state is not None else None),
                    }
                    # Re-enumerate per parent, per tied target — capture
                    # raw_valid (no tabu) and filtered_valid (post-tabu).
                    for parent_idx, st in enumerate(beam):
                        nm = get_non_masters(st.expr, target_sector)
                        if not nm:
                            continue
                        mw = max(((weight(k)[0], weight(k)[1])) for k in nm)
                        tied = [k for k in nm if (weight(k)[0], weight(k)[1]) == mw]
                        # rebuild indirect_cache for re-enumeration
                        ic = (_aux_to_result(st.aux_flat, env)
                              if st.aux_flat is not None else None)
                        if ic is None:
                            continue
                        dummy_subs = {k: {} for k in st.resolved_subs.keys()}
                        expr_fp_ = frozenset(st.expr.items())
                        ts_ = (tabu_dict.get(expr_fp_)
                               if tabu_dict is not None else None)
                        cu_p = st.aux_flat[0]
                        rid_p = st.aux_flat[2]
                        N_ = ibp_env.N_INDICES
                        _hist_p = (_sol_history.get(expr_fp_)
                                    if _sol_history is not None else None)
                        for tgt in tied:
                            raw_valid = _packed_cu.enumerate_valid_actions_with_indirect_cache_packed(
                                tgt, ic, st.resolved_subs,
                                env.ibp_t, env.li_t, env.shifts, 'subsector',
                                env._raw_eq_cache, _V7_REGISTRY, ibp_env.N_INDICES,
                            )
                            filtered = ([(op, dlt) for (op, dlt) in raw_valid
                                          if (tgt, op, dlt) not in ts_]
                                        if ts_ else list(raw_valid))
                            # AUDIT: for each tabu'd action, compute CURRENT
                            # sol_fp under THIS step's RS and compare to the
                            # sol_fp(s) when it was first tabu'd. Drift =
                            # current ≠ all historical sol_fps for that key.
                            audit_rows = []
                            if _hist_p is not None:
                                for (op, dlt) in raw_valid:
                                    key_ = (tgt, op, dlt)
                                    hist = _hist_p.get(key_, [])
                                    seed_p = tuple(tgt[i] + dlt[i] for i in range(N_))
                                    cu_idx_p = rid_p.get((op, seed_p))
                                    cur_sol_fp = None
                                    if cu_idx_p is not None:
                                        sol_p = solve_ibp_for(_v7_as_dict(cu_p[cu_idx_p]), tgt)
                                        if sol_p is not None:
                                            cur_sol_fp = hash(frozenset(sol_p.items()))
                                    hist_sol_fps = {sfp for _s, sfp in hist}
                                    drifted = (cur_sol_fp is not None
                                                and cur_sol_fp not in hist_sol_fps
                                                and len(hist) > 0)
                                    audit_rows.append({
                                        'op': op, 'delta': tuple(dlt),
                                        'current_sol_fp': cur_sol_fp,
                                        'history': hist,  # list of (step, sol_fp)
                                        'drifted': drifted,
                                    })
                            _stuck_payload['per_state_per_target'].append({
                                'parent_idx': parent_idx,
                                'target': tgt,
                                'n_raw': len(raw_valid),
                                'n_after_tabu': len(filtered),
                                'raw_valid': [(op, tuple(d)) for op, d in raw_valid],
                                'filtered': [(op, tuple(d)) for op, d in filtered],
                                'audit': audit_rows,
                            })
                    with open(_dump_stuck_path, 'wb') as _df:
                        pickle.dump(_stuck_payload, _df)
                    print(f'  [STUCK_DUMP] wrote forensic state → '
                          f'{_dump_stuck_path}', flush=True)
                except Exception as _e:
                    print(f'  [STUCK_DUMP] failed: {_e}', flush=True)
            if _real_stuck:
                break

        # Run model batched on all tasks (optionally chunked).
        # Chunking bounds the peak transient activation memory in the forward
        # pass — large unchunked batches leave hundreds of MB in glibc's free
        # list every step (the activations are freed but glibc retains pages).
        # Per-chunk prepare + forward + write into a preallocated probs_all,
        # with explicit `del` so each chunk's tensors are released before the
        # next. None / 0 / >= len(batch_data) means no chunking (original).
        _t = time.time() if _v5_prof else 0
        batch_data = []
        for parent_idx, target, valid in tasks:
            s = beam[parent_idx]
            batch_data.append((s.expr, s.resolved_subs, valid, target_sector, target))
        # Compute the global action width once so probs_all shape is stable.
        global_max_actions_eff = min(
            max(len(d[2]) for d in batch_data), max_actions)
        chunk_sz = model_batch_chunk if model_batch_chunk else len(batch_data)
        chunk_sz = max(1, min(chunk_sz, len(batch_data)))
        if _v5_prof:
            _p2['batch_prep'] += time.time() - _t
            _t = time.time()
        probs_all = torch.zeros(len(batch_data), global_max_actions_eff)
        for chunk_start in range(0, len(batch_data), chunk_sz):
            chunk = batch_data[chunk_start:chunk_start + chunk_sz]
            # Cap chunk's max_actions to the global so the slice assign below
            # has consistent width.
            b = prepare_batched_input_v5_dummy(chunk, device,
                                                max_actions=global_max_actions_eff)
            with torch.no_grad():
                _, chunk_probs = model(
                    b['expr_integrals'], b['expr_coeffs'], b['expr_mask'],
                    b['sub_keys'], b['sub_repl_ints'], b['sub_repl_coeffs'],
                    b['sub_repl_mask'], b['sub_mask'],
                    b['action_ibp_ops'], b['action_deltas'], b['action_mask'],
                    b['sector_mask'], b['target_integral'],
                )
            cn = chunk_probs.shape[1]
            probs_all[chunk_start:chunk_start + len(chunk), :cn] = chunk_probs
            del b, chunk_probs
        probs = probs_all
        # OFF-MANIFOLD CONFIDENCE PROBE (SAILIR_CONF_PROBE=1). Logs the model's
        # top score per (parent,target) task BEFORE beam selection, so it is not
        # biased by the sort picking confident states. Training states are all
        # on a valid reduction path; beam states after step ~1 are not. Compare
        # these against the on-manifold baseline measured on val states.
        if os.environ.get('SAILIR_CONF_PROBE') == '1':
            with torch.no_grad():
                _pm = probs.max(dim=1).values.tolist()
            if _pm:
                _sm = sorted(_pm)
                _n = len(_sm)
                print(f'  [CONFPROBE] step={step} n_tasks={_n} '
                      f'max={_sm[-1]:.6f} p90={_sm[int(0.9*(_n-1))]:.6f} '
                      f'median={_sm[_n//2]:.6f} p10={_sm[int(0.1*(_n-1))]:.6f} '
                      f'min={_sm[0]:.6f} '
                      f'frac>0.99={sum(v>0.99 for v in _sm)/_n:.3f}', flush=True)
        if _v5_prof:
            _p2['model_fwd'] += time.time() - _t

        # For each task, take top-K actions by prob, apply, generate candidates
        K = max(1, top_k if top_k is not None else beam_width // 2)
        _pick_dump_step = [] if _pick_dump_dir else None
        for ti, (parent_idx, target, valid) in enumerate(tasks):
            n_v = min(len(valid), probs.shape[1])
            row = probs[ti, :n_v].numpy()
            parent_state = beam[parent_idx]
            if _bounded_tabu_on:
                # Bounded tabu: block previously-tried (op,delta) for this
                # (expr_fp,target), but cap the effective block at CAP. When
                # more than CAP currently-valid actions are tabu'd, keep the
                # lowest-prob CAP blocked and RE-ALLOW the highest-prob overflow
                # (the model's current favourites get re-tried under evolved RS).
                _key_bt = (frozenset(parent_state.expr.items()), target)
                _tset = _bt_tabu.get(_key_bt)
                if _tset:
                    _blk = [i for i in range(n_v) if valid[i] in _tset]
                else:
                    _blk = []
                _n_blocked = len(_blk)
                if _n_blocked > _tabu_cap_eff:
                    _blk.sort(key=lambda i: row[i])      # ascending prob
                    _eff_block = set(_blk[:_tabu_cap])   # lowest-prob CAP
                else:
                    _eff_block = set(_blk)
                if _eff_block:
                    _avail = [i for i in np.argsort(-row) if i not in _eff_block]
                    if _avail:
                        top_idx = np.array(_avail[:K], dtype=int)
                    else:
                        # EXHAUSTED: every valid action is tabu'd. Cycle —
                        # re-allow everything and re-pick the top-K best, so a
                        # trapped task retries its best actions under the evolved
                        # RS rather than producing nothing (= baseline's stuck).
                        top_idx = np.argsort(-row)[:K]
                else:
                    top_idx = np.argsort(-row)[:K]
                # Record ALL picks into the tabu set (aggressive, like baseline).
                _ts2 = _bt_tabu.setdefault(_key_bt, set())
                for _ai in top_idx:
                    _ts2.add(valid[int(_ai)])
            else:
                # Take top-K by prob
                top_idx = np.argsort(-row)[:K]
                _n_blocked = -1
            if _pick_dump_step is not None:
                _V_d = _n_blocked
                _picked_d = tuple(sorted(
                    (op_, tuple(d_)) for (op_, d_)
                    in (valid[int(_a)] for _a in top_idx)))
                _pick_dump_step.append((
                    tuple(sorted(parent_state.expr.items())),
                    tuple(target), int(n_v), _picked_d, int(_V_d),
                    tuple((op_, tuple(d_), float(row[_i]))
                          for _i, (op_, d_) in enumerate(valid[:n_v]))))
            # Tabu is recorded LATER — after beam selection and macro-dedup
            # — for ONLY the actions that actually survived into the next
            # beam, not every top-K attempt.
            for ai in top_idx:
                ai = int(ai)
                op, delta = valid[ai]
                _t = time.time() if _v5_prof else 0
                result = apply_action_v5(
                    parent_state, target, op, delta, float(row[ai]),
                    env, target_sector, start_w12,
                    use_incremental_aux=use_incremental_aux,
                    lazy_rs=lazy_rs,
                )
                if _v5_prof:
                    _p3['apply'] += time.time() - _t
                    _ct['apply_calls'] += 1
                if result is None:
                    continue
                child, sol = result
                candidates.append(child)
                cand_metadata.append((parent_state, target, sol))

        if _pick_dump_step is not None:
            with open(f'{_pick_dump_dir}/picks_step{step}.pkl', 'wb') as _pf:
                pickle.dump(_pick_dump_step, _pf)

        if not candidates:
            if verbose:
                print(f'[v6 step {step}] no successful candidates — STUCK', flush=True)
            break

        # Beam selection
        _t_sort = time.time() if _v5_prof else 0
        # Build id() → metadata-index map so we can find each survivor's parent.
        meta_by_id = {id(c): i for i, c in enumerate(candidates)}
        if beam_sort == 'weight':
            candidates.sort(key=sort_weight)
            beam = candidates[:beam_width]
        elif beam_sort == 'mixed':
            half = beam_width // 2
            by_w = sorted(candidates, key=sort_weight)
            by_t = sorted(candidates, key=sort_totalweight)
            new_beam = []
            seen = set()
            for c in by_w:
                if len(new_beam) >= half:
                    break
                cid = id(c)
                if cid not in seen:
                    seen.add(cid)
                    new_beam.append(c)
            for c in by_t:
                if len(new_beam) >= beam_width:
                    break
                cid = id(c)
                if cid not in seen:
                    seen.add(cid)
                    new_beam.append(c)
            for c in by_w:
                if len(new_beam) >= beam_width:
                    break
                cid = id(c)
                if cid not in seen:
                    seen.add(cid)
                    new_beam.append(c)
            beam = new_beam
        elif beam_sort == 'prob':
            candidates.sort(key=sort_prob)
            beam = candidates[:beam_width]
        elif beam_sort == 'prob_nm':
            candidates.sort(key=sort_prob_nm)
            beam = candidates[:beam_width]
        else:
            raise ValueError(f'unknown beam_sort {beam_sort}')

        if _v5_prof:
            _p3['sort'] += time.time() - _t_sort

        # ── v6 early macro-dedup (Issue #11) ────────────────────────────
        # The original macro-dedup runs at the TOP of the next step. By
        # then we've already paid the cost of materializing aux_flat for
        # every duplicate child that's about to be thrown away. Doing the
        # dedup BEFORE materialization is bit-identical (dedup key depends
        # only on expr / max_w12 / n_non_masters / score — all of which
        # are unchanged by materialization) and saves the wasted work.
        # The next-step dedup becomes a no-op.
        if len(beam) > 1:
            early_groups = {}
            for s in beam:
                efp = frozenset(s.expr.items())
                cur = early_groups.get(efp)
                if cur is None or (s.max_w12, s.n_non_masters, -s.score) < \
                                  (cur.max_w12, cur.n_non_masters, -cur.score):
                    early_groups[efp] = s
            if len(early_groups) < len(beam):
                if verbose:
                    print(f'[v6 step {step}] early-dedup: {len(beam)} → '
                          f'{len(early_groups)} distinct expr (pre-materialize)',
                          flush=True)
                beam = list(early_groups.values())

        # Tabu: now that beam survivors are finalised, record (target, op,
        # delta) of EACH survivor's last action under its parent's expr_fp.
        # Only survivors get tabu'd — top-K attempts whose children were
        # rejected by beam selection are NOT tabu'd (they may produce
        # different outcomes in future visits with different RS context).
        if tabu_dict is not None and not _bounded_tabu_on:
            N_ = ibp_env.N_INDICES
            for c in beam:
                meta_i = meta_by_id.get(id(c))
                if meta_i is None:
                    continue
                parent_state, target, _sol = cand_metadata[meta_i]
                # Last action in child's path = (target, op, delta) applied
                target_p, op_p, delta_p = c.path[-1]
                expr_fp_p = frozenset(parent_state.expr.items())
                ts = tabu_dict.setdefault(expr_fp_p, set())
                ts.add((target_p, op_p, delta_p))
                if _sol_history is not None:
                    seed_p = tuple(target_p[i] + delta_p[i] for i in range(N_))
                    cu_p_ = parent_state.aux_flat[0]
                    rid_p_ = parent_state.aux_flat[2]
                    cu_idx_p = rid_p_.get((op_p, seed_p))
                    if cu_idx_p is not None:
                        sol_p = solve_ibp_for(_v7_as_dict(cu_p_[cu_idx_p]), target_p)
                        if sol_p is not None:
                            sol_fp_p = hash(frozenset(sol_p.items()))
                            _hist = _sol_history.setdefault(expr_fp_p, {})
                            _hist.setdefault((target_p, op_p, delta_p),
                                              []).append((step, sol_fp_p))

        # Post-selection survivors: materialize lazy_rs THEN attach aux in a
        # single pass so the original `c` is still in meta_by_id. Each survivor
        # is independent (beam[i] = f(c_i) using only read-only metadata/env).
        #
        # BLOCKED — naive fork is INCORRECT here: _attach_incremental_aux builds
        # new raw equations that REGISTER new integrals in _V7_REGISTRY. In a
        # forked worker those registrations land in the child's COW registry; the
        # returned PackedEq states carry ids the MAIN registry never saw, so the
        # next step's enumerate hits reg.get_tuple(unknown_id) -> IndexError.
        # (P1 is immune because its results are registry-independent (op,delta)
        # tuples.) The registry-merge/remap below makes the fork CORRECT and
        # bit-identical (FINAL REDUCTION IDENTICAL on (7,4)). BUT it is a MEASURED
        # NET LOSS and stays gated OFF: same-node A/B (node24, nw8+nthr8) was
        # SLOWER at every step (e.g. step29 97.1s on vs 69.4s off) and +13% peak
        # RSS (1877 vs 1659 MB). Cause: unlike P1's tiny (op,delta) outputs, each
        # P4 worker returns a full materialized state (resolved_subs + packed
        # aux_flat cu, ~1MB+), and pickling ~40 of those back + deserializing
        # serially in the main + the remap costs more than the 123s of attach_aux
        # we parallelized. Fork-IPC only pays when worker OUTPUT is small. To make
        # P4 fork win, the big arrays must go via SHARED MEMORY (ship handles, not
        # pickles) — not done. Default OFF; n_workers>1 runs P1-fork + P4-serial.
        _reg_probe = os.environ.get('SAILIR_REG_PROBE') == '1'
        if _reg_probe:
            _reg_p4_start = len(_V7_REGISTRY)
            _cu_total = sum(len(s.aux_flat[0]) for s in beam
                            if s.aux_flat is not None)
        if n_workers > 1 and os.environ.get('SAILIR_P4_FORK') == '1':
            _t_mat = time.time() if _v5_prof else 0
            # Registry-safe P4 fork. Each worker materializes a disjoint survivor
            # slice; attach_aux registers new integrals in the worker's COW
            # registry, so the returned PackedEq carry worker-local ids the main
            # never saw. To stay BIT-IDENTICAL (not just same-reduction), the main
            # reassigns global ids in the EXACT order serial would: serial
            # registers new integrals as survivors 0,1,2,... materialize, so we
            # capture each survivor's new tuples (in its registration order) and
            # replay get_id in beam-index order. That reproduces serial's id
            # values -> identical PackedEq sort order -> identical model scores.
            # Then relabel each worker's states local->global.
            _base_N = len(_V7_REGISTRY)

            def _mat_slice(indices):
                out = {}
                new_by_s = {}            # survivor idx -> its new tuples, in order
                for i in indices:
                    _b = len(_V7_REGISTRY)
                    c = beam[i]
                    ps, tg, sl = cand_metadata[meta_by_id[id(c)]]
                    if lazy_rs:
                        c = _materialize_lazy_rs(c, ps, tg, sl, start_w12)
                    if use_incremental_aux:
                        c = _attach_incremental_aux(
                            c, ps, tg, env, target_sector,
                            use_exprkeyed=use_exprkeyed,
                            iraws_window=iraws_window,
                            iraws_keep_first=iraws_keep_first,
                        )
                    out[i] = c
                    new_by_s[i] = _V7_REGISTRY._to_tuple[_b:]
                return out, new_by_s

            _mat_parts = _round_robin_parts(len(beam), n_workers)
            _results = _forkpool_map(_mat_slice, _mat_parts)
            # Pass 1: replay registration in beam-index order -> serial id values.
            _all_new = {}
            for _states, _new_by_s in _results:
                _all_new.update(_new_by_s)
            for _i in sorted(_all_new):
                for _T in _all_new[_i]:
                    _V7_REGISTRY.get_id(_T)
            # Pass 2: per worker, build local(base_N+k)->global remap from the
            # worker's tuples in its own processing order, then relabel + place.
            for _w, (_states, _new_by_s) in enumerate(_results):
                _wt = []
                for _i in _mat_parts[_w]:
                    _wt.extend(_new_by_s[_i])
                if _wt:
                    _remap = np.empty(len(_wt), dtype=np.int32)
                    for _k, _T in enumerate(_wt):
                        _remap[_k] = _V7_REGISTRY.get_id(_T)
                    for _i, _c in _states.items():
                        beam[_i] = _remap_state(_c, _base_N, _remap)
                else:
                    for _i, _c in _states.items():
                        beam[_i] = _c
            if _v5_prof:
                # Can't split mat_rs/attach_aux across the fork boundary; the
                # whole parallel P4 wall-clock lands under mat_rs.
                _p4['mat_rs'] = _p4.get('mat_rs', 0.0) + (time.time() - _t_mat)
        else:
            # === serial P4 — original code, unchanged ===
            _t_mat = time.time() if _v5_prof else 0
            _t_aux = 0.0
            for i, c in enumerate(beam):
                parent_state, target, sol = cand_metadata[meta_by_id[id(c)]]
                if lazy_rs:
                    c = _materialize_lazy_rs(c, parent_state, target, sol, start_w12)
                if use_incremental_aux:
                    _t1 = time.time() if _v5_prof else 0
                    c = _attach_incremental_aux(
                        c, parent_state, target, env, target_sector,
                        use_exprkeyed=use_exprkeyed,
                        iraws_window=iraws_window,
                        iraws_keep_first=iraws_keep_first,
                    )
                    if _v5_prof:
                        _t_aux += time.time() - _t1
                        _ct['attach_calls'] += 1
                beam[i] = c
            if _v5_prof:
                _p4['mat_rs'] = _p4.get('mat_rs', 0.0) + (time.time() - _t_mat - _t_aux)
                _p4['attach_aux'] += _t_aux
        if _reg_probe:
            # New integrals registered during P4 = the remap workload a fork would
            # have to reconcile. _cu_total = packed cu entries across survivors =
            # arrays that might need a remap/re-sort scan.
            _reg_p4_new = len(_V7_REGISTRY) - _reg_p4_start
            print(f'  [REGPROBE step {step+1}] reg {_reg_p4_start}->'
                  f'{len(_V7_REGISTRY)} (+{_reg_p4_new} new in P4) '
                  f'beam={len(beam)} cu_total={_cu_total} '
                  f'new_per_cu={_reg_p4_new/max(_cu_total,1):.4f}', flush=True)

        # Track best so far: use (max_w, n_non_masters) only — ignore score.
        # Score is cumulative negative log-prob, so the initial state always
        # has the "highest" score and would never be displaced by progress.
        def _progress_key(s):
            return (s.max_w12, s.n_non_masters,
                    -len(s.path))  # tiebreak: prefer longer path (more progress)

        best_in_beam = min(beam, key=_progress_key)
        if _progress_key(best_in_beam) < _progress_key(best_state):
            best_state = best_in_beam

        if verbose:
            mw_best = max_w12(best_in_beam.expr, target_sector)
            nm_best = best_in_beam.n_non_masters
            sz_rs = len(best_in_beam.resolved_subs)
            rs_vsz = sum(len(v) for v in best_in_beam.resolved_subs.values())
            # v6: count distinct exprs in beam to measure diversity gain.
            n_uniq_expr = len({frozenset(s.expr.items()) for s in beam})
            print(f'[v6 step {step+1:>3}] beam={len(beam)} '
                  f'uniq_expr={n_uniq_expr} '
                  f'best mw={mw_best} nm={nm_best} '
                  f'rs={sz_rs} rs_vsz={rs_vsz} '
                  f'expr={len(best_in_beam.expr)} '
                  f'cand={len(candidates)} '
                  f't_step={time.time()-t_step:.1f}s '
                  f't_total={time.time()-t0:.1f}s',
                  flush=True)
            if _v5_prof:
                p1_t = sum(_p1.values())
                p2_t = sum(_p2.values())
                p3_t = sum(_p3.values())
                p4_t = sum(_p4.values())
                total = p1_t + p2_t + p3_t + p4_t
                step_t = time.time() - t_step
                p1_parts = ' '.join(f'{k}={v:.2f}s' for k, v in _p1.items())
                p3_parts = ' '.join(f'{k}={v:.2f}s' for k, v in _p3.items())
                p2_parts = ' '.join(f'{k}={v:.2f}s' for k, v in _p2.items())
                p4_parts = ' '.join(f'{k}={v:.2f}s' for k, v in _p4.items())
                print(f'  PROF step={step+1} t_step={step_t:.2f}s '
                      f'P1={p1_t:.2f}s P2={p2_t:.2f}s P3={p3_t:.2f}s '
                      f'P4={p4_t:.2f}s residual={step_t-total:.2f}s', flush=True)
                print(f'    P1: {p1_parts}', flush=True)
                print(f'    P2: {p2_parts}', flush=True)
                print(f'    P3: {p3_parts}', flush=True)
                print(f'    P4: {p4_parts}', flush=True)
                print(f'    cts: ' + ' '.join(f'{k}={v}' for k, v in _ct.items()),
                      flush=True)
        # Explicit aux dump (SAILIR_DUMP_AUX_STEPS="10,25"): print the best
        # state's aux_flat = (cu, ubm, rid, iraws) — lengths, the per-cu-entry
        # term counts, and samples — so depth-keyed vs exprkeyed aux can be
        # compared directly. SEPARATE block (not inside the profiling guard).
        _dump_steps = os.environ.get('SAILIR_DUMP_AUX_STEPS')
        if _dump_steps and (step + 1) in {int(_x) for _x in _dump_steps.split(',') if _x}:
            _ax = best_in_beam.aux_flat
            if _ax is None:
                print(f'[AUXDUMP step {step+1}] best.aux_flat is None', flush=True)
            else:
                _cu, _ubm, _rid, _iraws = _ax
                _sizes = sorted((len(c) for c in _cu), reverse=True)
                print(f'[AUXDUMP step {step+1}] use_exprkeyed={use_exprkeyed} '
                      f'| len(cu)={len(_cu)} len(iraws)={len(_iraws)} '
                      f'len(rid)={len(_rid)} | cu #terms total={sum(_sizes)} '
                      f'max={_sizes[0] if _sizes else 0} top8={_sizes[:8]}',
                      flush=True)
                print(f'  iraws[:4] = {_iraws[:4]}', flush=True)
                if _cu:
                    _big = max(_cu, key=len)
                    print(f'  largest cu entry: {len(_big)} terms; first 4 items'
                          f' = {list(_big.items())[:4]}', flush=True)
                print(f'  rid[:3] = {list(_rid.items())[:3]}', flush=True)

        def _stream_dump_ckpt(path):
            # Stream the checkpoint ONE beam-state at a time so each state's
            # to_dict-materialized cu is freed before the next. memray showed
            # the whole-beam to_dict (was _serialize_beam_for_ckpt) at 5.2GB /
            # 59% of peak — the entire beam's packed cu materialized as dicts
            # simultaneously. Each state is an INDEPENDENT pickle frame (fresh
            # memo per pickle.dump) so `del sd` actually frees it; a single
            # shared Pickler would pin every object in its memo and save
            # nothing. Format: a meta dict (_streamed=True, n_states) then
            # n_states state-dicts in one file; the resume path detects
            # _streamed and reads n_states frames. Serialization-only change:
            # the live beam is untouched (s._asdict() + _aux_to_picklable both
            # create fresh objects), so trajectory/result stay bit-identical.
            with open(path, 'wb') as f:
                pickle.dump({
                    '_streamed': True,
                    'step': step + 1,
                    'n_states': len(beam),
                    'target_sector': target_sector,
                    'start_w12': start_w12,
                    'tabu_dict': _serialize_tabu_for_ckpt(),
                }, f)
                for s in beam:
                    sd = s._asdict()
                    if sd.get('aux_flat') is not None:
                        sd['aux_flat'] = _aux_to_picklable(sd['aux_flat'])
                    pickle.dump(sd, f)
                    del sd

        def _serialize_tabu_for_ckpt():
            # Convert {frozenset(expr_items): set((target, op, delta), ...)} to
            # a picklable list. frozenset+set don't have stable ordering on
            # disk but content is preserved; restoration rebuilds the same
            # structure (set membership is what matters for tabu lookup).
            if tabu_dict is None:
                return None
            return [
                (tuple(expr_fp), [(tuple(t), op, tuple(dlt))
                                  for t, op, dlt in entries])
                for expr_fp, entries in tabu_dict.items()
            ]

        # Full-memory probe (gated by SAILIR_MEMPROBE_FULL=/path/to/jsonl).
        # Samples every SAILIR_MEMPROBE_FULL_EVERY steps (default 10) using
        # pympler.asizeof for deep-size accounting of every persistent
        # structure. See scripts/eval/archive/memprobe_full.py.
        # End-of-step malloc_trim — release glibc free top after the per-step
        # transients have been freed. Single syscall; runs every step when
        # SAILIR_END_OF_STEP_TRIM=1.
        if _end_of_step_trim is not None:
            try:
                _end_of_step_trim(0)
            except Exception:
                pass

        if _memprobe_path:
            if _memprobe_steps_set is not None:
                _should_sample = (step + 1) in _memprobe_steps_set
            else:
                _should_sample = ((step + 1) >= _memprobe_start
                                   and (step + 1) % _memprobe_every == 0)
            if _should_sample:
                try:
                    import sys as _sys
                    _archive_dir = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), 'archive')
                    if _archive_dir not in _sys.path:
                        _sys.path.insert(0, _archive_dir)
                    import memprobe_full as _mpf
                    _mpf.measure(env, beam, tabu_dict, step + 1, _memprobe_path,
                                  model=model)
                except Exception as _e:
                    print(f'  [memprobe_full] error: {_e}', flush=True)

        # Focused memory breakdown. peak_rss = this step's transient peak (VmHWM
        # was reset at step start); rss = post-trim quiescent. Named structures
        # use ONE shared seen-set so they partition (no double count). The two
        # gaps separate the dominant unaccounted into its two possible causes.
        if _mem_bd and (step + 1) % _mem_bd_every == 0:
            # ── O(1) allocator-level accounting — NO object-graph walk. ────────
            # A heap walk (deep-size / gc.get_referents) needs a `seen` set of
            # one id PER LIVE OBJECT to dedup the traversal. This workload has
            # ~10^8 tiny int/set/dict objects, so that seen-set is *tens of GB*,
            # allocated every sample, larger than the heap it measures — that is
            # what ballooned rss to 146GB. So we NEVER walk objects here.
            #
            # smaps is the OS ground truth: heap+anon+file = rss EXACTLY (closes
            # by construction). malloc_info splits the glibc arena into
            # in-use/free/mmap. The CPython pymalloc arenas — where the millions
            # of small int/set/dict objects actually live — are anon bytes NOT in
            # glibc's mmap, so pymalloc_region = anon − glibc_mmap. Cost is O(1)
            # (a couple /proc reads + one malloc_info), so it cannot perturb the
            # run. It attributes rss to allocator buckets but NOT to python type/
            # source — for that, run a BOUNDED tracemalloc job (capped steps).
            _rss, _peak = _membd_mem()
            _heap, _anon, _filed = _membd_smaps()
            _rssm = _rss // 1024
            _peakm = _peak // 1024
            _heap_mb = _heap // 1024
            _anon_mb = _anon // 1024
            _file_mb = _filed // 1024
            _mi = _membd_mallinfo()
            if _mi is not None:
                _g_inuse = _mi['uordblks'] // 10**6
                _g_free = _mi['fordblks'] // 10**6
                _g_mmap = _mi['hblkhd'] // 10**6
            else:
                _g_inuse = _g_free = _g_mmap = -1
            _pyrest_mb = max(_anon_mb - max(_g_mmap, 0), 0)
            _close = _heap_mb + _anon_mb + _file_mb
            print(f'[MEMBD step {step+1}] peak_rss={_peakm}MB rss={_rssm}MB '
                  f'| smaps heap={_heap_mb}MB anon={_anon_mb}MB file={_file_mb}MB '
                  f'(sum={_close}MB={100*_close//max(_rssm,1)}% of rss) '
                  f'| glibc[inuse={_g_inuse} free={_g_free} mmap={_g_mmap}]MB '
                  f'pymalloc+torch+numpy(anon-glibcmmap)={_pyrest_mb}MB '
                  f'| n_beam={len(beam)} n_cand={len(candidates)} '
                  f'n_cache={len(env._raw_eq_cache)} '
                  f'n_bt={len(_bt_tabu) if _bt_tabu else 0}',
                  flush=True)

        # Checkpoint (rolling, overwrites) — streamed one state at a time.
        if ckpt_path and (step + 1) % ckpt_every == 0:
            _stream_dump_ckpt(ckpt_path)
            if verbose:
                print(f'  [ckpt] wrote {ckpt_path} at step {step+1}', flush=True)
        # KEPT versioned checkpoints every SAILIR_KEEP_CKPT_EVERY steps (not
        # overwritten) — lets us reconstruct the persistent state at the peak
        # step after the fact, correlated with the per-step MEMBD peak_rss log.
        _keep_every = int(os.environ.get('SAILIR_KEEP_CKPT_EVERY', '0'))
        if ckpt_path and _keep_every > 0 and (step + 1) % _keep_every == 0:
            _kp = f'{ckpt_path}.keep_step{step + 1:05d}'
            _stream_dump_ckpt(_kp)
            if verbose:
                print(f'  [keep-ckpt] wrote {_kp}', flush=True)
        # Per-step thick checkpoint (for bit-identical incremental verification)
        if ckpt_every_step and ckpt_path:
            step_path = f'{ckpt_path}.step{step + 1:04d}'
            _stream_dump_ckpt(step_path)

    return beam, best_state


# ============================================================================
# Path replay for final answer
# ============================================================================

def replay_full_expr(start_expr, path, env):
    """Replay path against the FULL start_expr (no stripping) to recover the
    complete final expression including all passenger spillover.
    """
    expr = dict(start_expr)
    subs = {}  # raw subs for replay
    # UNSTRIPPED raws for replay: the search's env._raw_eq_cache may hold
    # sub-weight-STRIPPED raws (mod-lower-weight), which would give an incomplete
    # final_expr. Replay needs the FULL raws, so recompute them into a small
    # local cache (just the path's ~len(path) raws), independent of the search.
    _full_raw_cache = {}
    for (target, ibp_op, delta) in path:
        seed = tuple(target[i] + delta[i] for i in range(ibp_env.N_INDICES))
        _rk = (ibp_op, seed)
        raw = _full_raw_cache.get(_rk)
        if raw is None:
            raw = ibp_env.get_raw_equation(env.ibp_t, env.li_t, ibp_op, seed)
            _full_raw_cache[_rk] = raw
        # Apply current subs to raw to get cached
        from sailir.ibp_env import apply_all_substitutions
        cached = apply_all_substitutions(raw, subs)
        if target not in cached or cached[target] == 0:
            return None
        sol = solve_ibp_for(cached, target)
        if sol is None:
            return None
        subs[target] = sol
        # Apply substitution to expr (no stripping — full)
        from sailir.ibp_env import apply_substitution as _apply
        expr = _apply(expr, target, sol)
    return expr, subs


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--topology', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--integral', required=True,
                   help='Comma-separated indices for starting integral')
    p.add_argument('--prime', type=int, default=1009)
    p.add_argument('--packed-rs', action='store_true',
                   default=(os.environ.get('SAILIR_PACKED_RS', '0') == '1'),
                   help='Stage 3b: store State.resolved_subs values as PackedEq '
                        '(int32/int16) instead of dicts. Bit-identical; lower '
                        'memory. Also enabled by SAILIR_PACKED_RS=1.')
    p.add_argument('--beam-width', type=int, default=40)
    p.add_argument('--top-k', type=int, default=None,
                   help='Number of model-top actions tried per (parent,target)'
                        ' per step. Default = beam_width // 2 = 20.')
    p.add_argument('--max-steps', type=int, default=2000)
    p.add_argument('--max-actions', type=int, default=900)
    p.add_argument('--beam-sort', default='mixed')
    p.add_argument('--paper-masters-only', action='store_true', default=True)
    p.add_argument('--no-paper-masters-only', dest='paper_masters_only',
                   action='store_false')
    p.add_argument('--output', default=None)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--ckpt-every', type=int, default=50)
    p.add_argument('--resume-from', default=None,
                   help='Path to a ckpt or result pkl to resume the beam from')
    p.add_argument('--tabu', action='store_true',
                   help='Enable per-expr action tabu (block re-pick of same '
                        '(target,op,delta) from same expr)')
    p.add_argument('--no-incremental-aux', dest='use_incremental_aux',
                   action='store_false', default=True,
                   help='Disable aux_flat incremental update — rebuild every '
                        'step from scratch (slower, for verification only)')
    p.add_argument('--no-exprkeyed', dest='use_exprkeyed',
                   action='store_false', default=True,
                   help='Disable bounded-anchor exprkeyed delta — fall back to '
                        'compute_indirect_substituted_incremental (slower at '
                        'high |RS|, for verification only)')
    p.add_argument('--ckpt-every-step', action='store_true',
                   help='Write a thick per-step checkpoint after every step '
                        '(format: <ckpt_path>.stepNNNN). For bit-identical '
                        'verification of incremental-aux runs.')
    p.add_argument('--iraws-window', type=int, default=None,
                   help='If set, keep iraws anchored on the most recent N '
                        'sub_ints (recency window).')
    p.add_argument('--iraws-keep-first', type=int, default=None,
                   help='If set, keep iraws anchored on the FIRST N sub_ints '
                        '(bootstrap anchors). Empirically, drain-critical '
                        'Phase-1b hooks come from these. Combine with '
                        '--iraws-window for first-N ∪ last-M coverage.')
    p.add_argument('--no-lazy-rs', dest='lazy_rs', action='store_false',
                   default=True,
                   help='Disable LAZY_RS optimization (apply add_sub_to_resolved '
                        'eagerly for all candidates instead of only survivors). '
                        'Slower; for verification only.')
    p.add_argument('--model-batch-chunk', type=int, default=8,
                   help='If set, split the model forward batch into chunks of '
                        'this many rows. Bounds transient activation memory '
                        '(glibc retains free pages from large unchunked '
                        "forwards). None = no chunking (original behavior).")
    # OPTIMAL PRODUCTION SETTING: --n-threads 8 --n-workers 8 (on an 8-core
    # request). Measured same-node (7,4): 8/8 = ~319s vs 4/4 = ~438s (+37%).
    # Breakdown of the loss at 4: ~85% is the model (P2, scales ~1.8x with
    # threads) and ~15% enumerate (P1). Since the 8-core request is dictated by
    # the model threads and is reserved for the whole job, run 8 workers too —
    # fewer would just idle reserved cores during enumerate. Defaults stay 1/1
    # for reproducible single-core runs and quick login sanity checks.
    p.add_argument('--n-threads', type=int, default=1,
                   help='torch threads for the model phase (P2). PRODUCTION: 8. '
                        'The model is ~40%% of the run and scales ~1.8x from 4->8 '
                        'threads. Sets request_cpus.')
    p.add_argument('--n-workers', type=int, default=1,
                   help='processes for the enumerate phase (P1). 1 = serial '
                        '(original path, zero pooling overhead). >1 forks this '
                        'many workers per step over disjoint parents. Independent '
                        'of --n-threads (model, P2). PRODUCTION: set equal to '
                        '--n-threads (8) — those cores are reserved for the whole '
                        'job, so fewer workers just idle them during enumerate.')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    # Memprobe init: smaps-based attribution. tracemalloc is NOT started
    # by default — tracing every Python allocation slows the whole program
    # 5-10×, even when we only sample at a few specific steps. Set
    # SAILIR_MEMPROBE_FULL_TRACEMALLOC=1 to opt in for Python-heap detail.
    _memprobe_mod = None
    if os.environ.get('SAILIR_MEMPROBE_FULL'):
        if os.environ.get('SAILIR_MEMPROBE_FULL_TRACEMALLOC') == '1':
            import tracemalloc as _tm
            _tm.start(int(os.environ.get(
                'SAILIR_MEMPROBE_FULL_TRACEMALLOC_FRAMES', '5')))
        try:
            open(os.environ['SAILIR_MEMPROBE_FULL'], 'w').close()
        except OSError:
            pass
        import sys as _sys
        _archive_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'archive')
        if _archive_dir not in _sys.path:
            _sys.path.insert(0, _archive_dir)
        import memprobe_full as _memprobe_mod
        _memprobe_mod.record_milestone('rss_at_import_kb')

    torch.set_num_threads(args.n_threads)
    # torch's INTER-op pool is SEPARATE from set_num_threads (intra-op) and
    # defaults to the node core count -> on heavy batches it spills past
    # request_cpus (the intra-op pool already uses all n_threads, leaving no
    # headroom). Cap it to 1 whenever we are multi-threaded/forked, matching the
    # _cap_incidental_threads philosophy. SAILIR_CAP_INTEROP=0 forces it off.
    if (os.environ.get('SAILIR_CAP_INTEROP', '1') != '0'
            and (args.n_threads > 1 or args.n_workers > 1)):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass   # already initialized / parallel work started

    print(f'== v5 beam search ==', flush=True)
    print(f'  topology      = {args.topology}', flush=True)
    print(f'  model         = {args.model}', flush=True)
    print(f'  integral      = {args.integral}', flush=True)
    print(f'  beam_width    = {args.beam_width}', flush=True)
    print(f'  max_steps     = {args.max_steps}', flush=True)
    print(f'  max_actions   = {args.max_actions}', flush=True)
    print(f'  beam_sort     = {args.beam_sort}', flush=True)

    topology = Topology.from_dir(args.topology)
    init_from_topology(topology)
    assert topology.n_denominators == _TC_N_DEN, (
        f"topology has {topology.n_denominators} denominators but "
        f"SAILIR_TOPOLOGY={os.environ.get('SAILIR_TOPOLOGY', 'pentagonbox')!r} "
        f"configures {_TC_N_DEN} — set SAILIR_TOPOLOGY to match --topology")
    set_prime(args.prime)
    set_paper_masters_only(args.paper_masters_only)
    env = IBPEnvironment()
    global _V7_REGISTRY, _V7_PACKED_RS_CACHE, _PACKED_RS
    _V7_REGISTRY = IntegralRegistry()
    _V7_PACKED_RS_CACHE = {}
    _PACKED_RS = args.packed_rs
    if _PACKED_RS:
        print('  packed-rs: State.resolved_subs values stored as PackedEq',
              flush=True)
    if _memprobe_mod is not None:
        _memprobe_mod.record_milestone('rss_after_env_kb')

    # prime is REQUIRED (no default since the 2026-07-15 bug fix): take it from
    # the checkpoint's own training args, falling back to the CLI prime.
    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    _cka = ck.get('args') or {}
    # Pick the class the checkpoint was TRAINED with. Defaults to the historical
    # IBPActionClassifier so every pre-existing checkpoint loads exactly as before;
    # a 'nosubs' checkpoint has no subs_enc.* keys and would fail load_state_dict
    # against the full class.
    _variant = _cka.get('model_variant', 'full')
    # score_activation decides forward()'s SECOND return value, which is what
    # this beam consumes as action_prob. 'sigmoid' for bce-trained models --
    # softmax would divide by sum_j exp(z_j) over the action set and reimpose
    # the action-count dependence those models are trained without.
    _act = _cka.get('score_activation', 'softmax')
    _mkw = dict(
        prime=_cka.get('prime', args.prime),
        n_indices=topology.n_indices,
        n_denominators=topology.n_denominators,
        n_ibp_ops=topology.n_actions,
    )
    for _k in ('embed_dim', 'n_heads', 'n_expr_layers', 'n_cross_layers'):
        if _k in _cka:
            _mkw[_k] = _cka[_k]
    if _variant == 'nosubs':
        from sailir.classifier_nosubs import IBPActionClassifierNoSubs
        model = IBPActionClassifierNoSubs(score_activation=_act, **_mkw)
    else:
        model = IBPActionClassifier(**_mkw)
    print(f'  model_variant = {_variant}  score_activation = {_act}  '
          f'prime = {_mkw["prime"]}', flush=True)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    if _memprobe_mod is not None:
        _memprobe_mod.record_milestone('rss_after_model_kb')
        _memprobe_mod.record_model(model)

    # Strip surrounding quotes that may survive condor argv parsing
    integral_str = args.integral.strip("'").strip('"')
    start_int = tuple(int(x) for x in integral_str.split(','))
    start_w = weight(start_int)
    start_w12 = (start_w[0], start_w[1])
    global _START_TOTAL_KEY
    _START_TOTAL_KEY = _target_key(start_int)   # for SAILIR_SUCCESS_TOTAL=1
    print(f'  start integral weight = {start_w}', flush=True)
    print(f'  active threshold      = (w1,w2) >= {start_w12}', flush=True)
    print(f'  SINGLE-STEP reducer; success criterion = '
          f'{"TOTAL-weight active bucket drained" if _SUCCESS_TOTAL else "(w1,w2) active bucket drained"}',
          flush=True)
    # Strip sub-weight from the CACHED raws used by the search (cu built mod-
    # lower-weight). Must be BEFORE any raw is cached. Replay uses the unstripped
    # get_raw_equation, so final_expr is unaffected. SAILIR_STRIP_RAWS=0 disables.
    if os.environ.get('SAILIR_STRIP_RAWS', '1') != '0':
        ibp_env.set_raw_strip_threshold(start_w12)
        print(f'  raw-strip: cached raws stripped to (w1,w2) >= {start_w12}', flush=True)

    target_sector = tuple(get_sector_mask(start_int))
    print(f'  target_sector = {target_sector}', flush=True)

    start_expr = {start_int: 1}

    beam, best_state = beam_search_v5(
        env, model, start_expr, target_sector, start_w12,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        device=args.device,
        beam_sort=args.beam_sort,
        max_actions=args.max_actions,
        ckpt_path=args.ckpt,
        ckpt_every=args.ckpt_every,
        resume_from=args.resume_from,
        tabu=args.tabu,
        use_incremental_aux=args.use_incremental_aux,
        use_exprkeyed=args.use_exprkeyed,
        ckpt_every_step=args.ckpt_every_step,
        iraws_window=args.iraws_window,
        iraws_keep_first=args.iraws_keep_first,
        lazy_rs=args.lazy_rs,
        model_batch_chunk=args.model_batch_chunk,
        top_k=args.top_k,
        n_workers=args.n_workers,
    )

    print(f'\n== DONE — beam size {len(beam)} ==', flush=True)
    if os.environ.get('SAILIR_P4_TPROBE') == '1':
        _tp = _packed_cu._TP
        _tot = _tp['total'] or 1.0
        _glue = _tp['total'] - _tp['subone'] - _tp['phaseB']
        print(f"[TPROBE attach_aux split over {_tp['n_calls']} calls] "
              f"total={_tp['total']:.1f}s | "
              f"subone(nogil-mergeable)={_tp['subone']:.1f}s "
              f"({100*_tp['subone']/_tot:.1f}%) | "
              f"phaseB(raw-build/registry,GIL)={_tp['phaseB']:.1f}s "
              f"({100*_tp['phaseB']/_tot:.1f}%) | "
              f"other-glue(GIL)={_glue:.1f}s ({100*_glue/_tot:.1f}%) | "
              f"n_subone={_tp['n_subone']} "
              f"avg_terms={_tp['sub_terms']/max(_tp['n_subone'],1):.1f}",
              flush=True)
        _nz = _tp['n_subone'] - _tp['n_noop']
        print(f"[TPROBE noop split] n_subone={_tp['n_subone']} | "
              f"NO-OP (sub_id absent, wasted visit)={_tp['n_noop']} "
              f"({100*_tp['n_noop']/max(_tp['n_subone'],1):.1f}%) | "
              f"real substitutions={_nz} "
              f"({100*_nz/max(_tp['n_subone'],1):.1f}%) | "
              f"avg_terms: noop={_tp['noop_terms']/max(_tp['n_noop'],1):.1f}",
              flush=True)
    print(f'best state: n_non_masters={best_state.n_non_masters} '
          f'mw={max_w12(best_state.expr, target_sector)} '
          f'path_len={len(best_state.path)} '
          f'score={best_state.score:.4f}', flush=True)
    # Peak RSS (kernel-tracked, monotonic over the process lifetime).
    try:
        with open('/proc/self/status') as _ps:
            for _line in _ps:
                if _line.startswith('VmHWM:'):
                    _peak_kb = int(_line.split()[1])
                    print(f'peak_rss_kb={_peak_kb} ({_peak_kb // 1024} MB)',
                          flush=True)
                    break
    except OSError:
        pass

    if _is_success(best_state, target_sector):
        print('SUCCESS — single-step reduction done: active bucket drained '
              '(start_int reduced one weight level -> passenger expansion)', flush=True)
    else:
        print('INCOMPLETE — active bucket still has non-masters', flush=True)

    if args.output:
        # Replay full expr (with passenger) for the best state
        print('Replaying path for full final expression...', flush=True)
        full_result = replay_full_expr(start_expr, best_state.path, env)
        out = {
            'best_state': best_state._asdict(),
            'beam': [s._asdict() for s in beam],
            'start_integral': start_int,
            'start_w12': start_w12,
            'target_sector': target_sector,
            'full_expr_replay': full_result[0] if full_result else None,
            'full_subs_replay': full_result[1] if full_result else None,
        }
        with open(args.output, 'wb') as f:
            pickle.dump(out, f)
        print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
