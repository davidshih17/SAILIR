#!/usr/bin/env python
# =========================================================================
# BEAM SEARCH V8 (2026-08-03): fork of beam_search_v7 with ONE change —
# per-step TARGETING selects the SINGLE highest integral in the workers'
# FULL total order (_target_key: sector rank when SAILIR_SECTOR_RANK=1,
# then (r,s), then |abs| lex), exactly like the training-data generator
# and the truth engine. v7 targeted ALL (r,s)-tied integrals per state
# (no lex tiebreak) — the long-flagged train/inference target-ordering
# mismatch ("Alignment is the next step"), measured 2026-08-02 to scatter
# the beam across untrained sibling targets on dot-heavy sectors.
# =========================================================================
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
  - --beam-sort {weight,mixed,prob,dual,toptotal}
  - any() termination on first state with nm=0
"""
import argparse
import math
import operator
import os
import pickle
import re
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
     'aux_flat',   # (cached_unique, union_bms, raw_id_to_idx, indirect_raws)
                   # or None to mean "rebuild from scratch on use"
     'lane'],      # DUAL beam only: which sub-beam this state belongs to.
                   # 0 = weight-sorted lane, 1 = prob-sorted lane. Inherited by
                   # children, so a lane's population descends only from itself
                   # and the two searches never compete for the same slots.
                   # Ignored (always 0) in every other beam_sort mode.
)
State_v5.__new__.__defaults__ = (None, None, None, 0)  # max_w12, total_w12, aux_flat, lane


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

# ---------------------------------------------------------------------------
# MODEL-FREE ACTION SCORING (measurement only; default OFF = unchanged behavior)
#
# The model's dominant role in the search is a ~1000:1 filter: top_idx =
# argsort(-row)[:K] keeps K=20 of ~9700 valid actions. Measured on the truth
# recordings, sorting those same actions by sum|seed| puts a CERTIFIED-good
# action at rank 0 half the time and inside the top 12 in 90% of steps -- so a
# one-line sort may do most of that filtering job. This switch replaces the
# per-action score with a model-free one so the claim can be tested end-to-end
# in search rather than only offline.
#
# 'model'   (default) -- untouched, row stays exactly the model's probabilities
# 'sumseed' -- score = -sum|seed|, seed = target + delta (the heuristic above)
# 'random'  -- the FLOOR. If random top-K solves nearly as much, the search is
#              easy and neither model nor heuristic is doing the work.
#
# Determinism: 'random' is seeded per (target, n_valid) with crc32, NOT Python's
# hash() -- hash() is salted per process, which would make runs unreproducible
# and silently break the bit-reproducibility the workers rely on.
# ---------------------------------------------------------------------------
# SAILIR_SCORE = local (default, unchanged) | cumulative.
# 'cumulative' makes State.score the running sum of log-probs along the path
# instead of only the last action's. See the note at the assignment site in
# apply_action_v5 for why the usual "penalises long lineages" objection does
# not apply (the beam advances in lockstep, so same-step states have equal
# path lengths -- verified via restart_offsets empty in 100/100 runs).
# ── TRUTH-TRACE ────────────────────────────────────────────────────────────
# SAILIR_TRUTH_TRACE=<recording.jsonl> follows the truth reduction alongside the
# search and records, step by step, whether the truth state is still inside the
# beam -- and when it is not, WHY.
#
# The question this answers cannot be answered from the ordinary logs. A run
# that ends up taking 3x the truth path length has left the truth trajectory
# somewhere, but "nm went up" does not say whether the beam never generated the
# right child or generated it and then culled it. Those need opposite fixes:
#
#   MODE A -- the truth state was in the beam, but the truth ACTION was not
#             among the top-K expanded from it. The right move was never even a
#             candidate. Fix: larger or adaptive K.
#   MODE B -- the truth child WAS a candidate and lost the sort. Fix: the sort
#             key, the lane split, or the beam width. The trace records its
#             rank and how far it missed the cutoff by, so the size of the fix
#             is measurable rather than guessed.
#
# Matching is by expression fingerprint: a beam state that has taken exactly
# the truth actions has exactly the truth expression, so equality of
# frozenset(expr.items()) at the same depth identifies it. Writing is
# append-only JSON lines and touches nothing the search reads, so behaviour is
# bit-identical with the variable unset -- and it is only worth paying for on a
# diagnostic run, since fingerprinting every candidate costs a pass over the
# candidate list.
# ---- CLOSURE-LIBRARY TRACE (SAILIR_LIB_TRACE=<lib json>) -----------------
# Asks, per step: are ANY beam states still inside the truth closure library?
#
# This replaces truth-PATH tracking, which was the wrong question twice over.
# (1) It followed the single LEX row, one arbitrary tie-break among ~17.9
#     equally-correct closure rows per step, so it punished the beam for
#     holding a different, equally-reducing state. (2) It matched path
#     prefixes, and a state is (expr, resolved_subs) -- prefix matching
#     conflates distinct states and produced three retracted results.
#
# The library is a SET of (op, seed) actions -- the dependency closure of the
# target rule -- and consuming it in ANY order succeeds (first/last/random
# reach success in 338/352/352 steps). So membership is PATH-INDEPENDENT: a
# state is inside iff every action on its path came from the set. No matching,
# no tie-break, no conflation.
_LIB_TRACE = os.environ.get('SAILIR_LIB_TRACE')
_LIB = None
if _LIB_TRACE:
    with open(_LIB_TRACE) as _lf:
        _LIB = {(int(r[0]), tuple(int(x) for x in r[1:]))
                for r in __import__('json').load(_lf)['library']}


_LIB_PARENT = []          # per-step: one entry per library-consistent parent
# Every parent, tagged library or not, with AVAIL-FREE shape stats only:
# top_prob and entropy. `conc = top_prob*avail` is undefined for a non-library
# state (no correct set exists off the recorded reduction), so the comparable
# quantities are the raw concentration measures.
_ALL_PARENT = []


def _in_library(st):
    """Every action on this state's path drawn from the closure library.

    path entries are (target, ibp_op, delta); the library stores (op, seed)
    with seed = target + delta, so the two are compared in seed space.
    """
    for _t, _op, _d in st.path:
        if (int(_op), tuple(int(a) + int(b) for a, b in zip(_t, _d))) not in _LIB:
            return False
    return True


_TRUTH_TRACE = os.environ.get('SAILIR_TRUTH_TRACE')
_TRACE_OUT = os.environ.get('SAILIR_TRACE_OUT', 'truth_trace.jsonl')
_truth_fp = []      # per truth step: frozenset(expr.items())
_truth_actions = []  # the truth walk's action sequence
_frontier = -1      # furthest truth index the beam has ever held
_trace_fh = None

# SOL-FINGERPRINT TABU (SAILIR_SOL_TABU=1).
# The existing tabu keys on (expression, target) and bans an (op, delta) pair
# outright. But a state is (expr, resolved_subs): the SAME (op, delta) resolved
# through a different substitution store yields a DIFFERENT substitution, so an
# action tried by one lineage is not the same move for another that merely
# shares the expression. Measured: at depth 3 of 2,1,0 this banned the truth
# action outright -- 33 of 242 candidates blocked, the correct one among them,
# and it never reached the model.
#
# The fix keys on what the action actually DID: the fingerprint of the resolved
# substitution. That is the one object distinguishing two states sharing an
# expression, it is computed anyway when the action is applied (free to record),
# and it is a single integer -- no snapshot of the store, which grows
# unboundedly along a lineage.
#
# Cost is bounded: the cheap (op, delta) key stays as a pre-filter, and the
# substitution is re-derived ONLY for candidates it blocks (33 of 242 here),
# never for the full list. A set, not a list, so it is bounded by the number of
# genuinely distinct substitutions rather than by visit count.
# LINEAGE-SCOPED TABU (SAILIR_LINEAGE_TABU=1).
# Today a ban written under (expression, target) applies to EVERY later state
# with that expression, whatever its lineage. Measured on 2,1,0 depth 3: the
# entry that banned the truth action was written by a state at depth 1 that is
# NOT an ancestor of the truth state (writers=1, ancestor_of_me=False). So the
# ban blocked a state that had never tried the action, on behalf of a branch it
# is not descended from -- which is not what tabu is for. Tabu exists to stop a
# search re-treading ITS OWN steps.
#
# Scoping by ancestry keeps that purpose and drops the cross-lineage damage. To
# avoid storing whole paths (they grow to thousands of entries), each writer is
# recorded as (path_length, hash(path)); a ban applies to state P iff some
# writer (L, H) has L <= len(P) and hash(P[:L]) == H -- i.e. the writer sits on
# P's ancestry. Cost is paid only for actions the cheap key already blocked.
# PROB FLOOR (SAILIR_PROB_FLOOR=x): drop actions the model rates below x
# before they can enter the beam.
#
# Motivation, measured at 2,1,0 step 3 under the stock lexicographic key: of the
# 113 candidates that outranked the truth child on max_w12 and took its beam
# slot, only 18 were rated ABOVE it by the model -- the median sat at p ~ 1e-5.
# The beam was spending 40 slots largely on moves the model considers
# near-impossible, purely because their w2 was one lower. A floor removes those
# without touching the sort, so no exchange rate has to be tuned and the
# threshold is defined on the model's own scale rather than relative to a
# candidate pool that shifts every time the sort changes.
#
# The top-1 action is always kept, so a state can never be left with nothing to
# expand.
# RESERVED PROB SLOTS (SAILIR_TOPN_PROB=n): give the model's n highest-scoring
# candidates guaranteed places, then fill the rest of the beam by the ordinary
# lex weight sort.
#
# Motivation: at 2,1,0 step 3 the truth child ranked 21st of 260 by model
# probability -- and the dual beam's prob lane holds exactly 20. It missed a
# guaranteed slot by one. Unlike an exchange rate between w2 and log p, this
# needs no tuning against a candidate pool that shifts whenever the sort
# changes: n is a slot count, and the model's ordering decides who fills it.
_TOPN_PROB = int(os.environ.get('SAILIR_TOPN_PROB', '0') or 0)
_PROB_FLOOR = float(os.environ.get('SAILIR_PROB_FLOOR', '0') or 0)
_RANK_SCORE = os.environ.get('SAILIR_RANK_SCORE') == '1'
# THE BEAM COMPARES LOG-PROBS ACROSS PARENTS, WHICH IS NOT A VALID COMPARISON.
# Each parent's row is a softmax over ITS OWN action set -- different sizes
# (70 to 2400 on one walk) and different difficulty -- so a parent where the
# model is sharp wins every slot from a parent where it is diffuse, regardless
# of correctness. Measured on 2,1,0 / soft_ep10 at the step the beam loses the
# closure library: the BEST library candidate scored -2.245 (p=0.106) while
# non-library candidates ran to -0.001 (p=0.999), and 40 of the latter cleared
# the -2.531 cutoff. Meanwhile val_anyhit20 = 0.995-0.999, i.e. a correct
# action is essentially ALWAYS in the parent's own top-20. Both are true: rank
# WITHIN a parent is excellent, absolute probability ACROSS parents is not
# comparable. The beam is a confidence magnet -- it fills with descendants of
# parents the model happens to be sharp about, and sharpness tracks how easy
# the state is, not whether the path is right.
#
# SAILIR_REL_SCORE=1 divides by the parent's own max, so the stored score
# becomes log(p_i / p_max): rank-1 from EVERY parent is 0, and within-parent
# gaps survive. Removes the cross-parent scale problem while keeping the
# within-state judgement anyhit20 says the model is good at.
_REL_SCORE = os.environ.get('SAILIR_REL_SCORE') == '1'
_RESERVE_PARENT = os.environ.get('SAILIR_RESERVE_PARENT') == '1'
# SAILIR_W1_FIRST=1 -- sort by w1 (the DOT COUNT) first, then by probability.
# Splits the difference between `prob` and `weight`, using ONLY the component
# that actually discriminates.
#
# MEASURED on soft_ep18, target 2,1,0, first 8 steps, 3339 candidates:
#   w1 > start_w1 :  0 of 1382 LIBRARY candidates  (0.0%)
#                  917 of 1957 NON-library         (46.9%)
# A perfect one-way separator -- w1 above the start means the dot count has
# INCREASED above the start integral, which cannot be on the reduction path.
# Deprioritising them lifts the library fraction 0.414 -> 0.571 (1.38x) every
# step, and r is what compounds.
#
# WHY NOT `weight` SORT: its key is (w1, w2, nm, -score). Measured, w2 and nm
# do NOT separate the two classes at all -- nm quantiles are [1,5,8,11,18] for
# library and [1,5,8,10,17] for non-library, identical. So `weight` pays for
# one good key with two useless ones ahead of the model. This uses w1 alone.
#
# WHY A PREFERENCE AND NOT A FILTER: if the w1<=start pool ever empties the
# beam can still fall back, where a hard drop would starve it.
_W1_FIRST = os.environ.get('SAILIR_W1_FIRST') == '1'
# SAILIR_W1_DROP=1 -- FILTER form of the same separator: discard candidates
# whose w1 exceeds the START integral's w1, leaving `prob` as the PRIMARY sort.
#
# Distinct from W1_FIRST, which SORTS by w1 and thereby demotes probability to
# a secondary key -- measured 20 (ep10) and 31 (ep18) against baselines 52 and
# 60. Eleven variants in, the consistent pattern is that ANYTHING outranking
# the model's probability loses, because that ranking supplies the whole ~1.5x
# per-step enrichment holding r near 1. This drops provably-wrong candidates
# WITHOUT touching the ordering of what remains.
#
# Sound because the separation is categorical, not distributional: 0 of 1382
# library candidates had w1 > start_w1, against 917 of 1957 non-library.
_W1_DROP = os.environ.get('SAILIR_W1_DROP') == '1'
# SAILIR_ENT_BONUS=lam adds lam * H(parent) in log space, i.e. favours children
# of HIGH-ENTROPY parents. Unlike REL_SCORE this can express "this parent is
# suspiciously peaked" rather than merely normalising every parent alike --
# worth separating because confidence is NOT always wrong (at step 50 the best
# library candidate carried p~1.0 and was correct), so a flat anti-confidence
# penalty would punish that too.
_ENT_BONUS = float(os.environ.get('SAILIR_ENT_BONUS', '0') or 0)
# SAILIR_ENT_CUT=x -- HARD cull: refuse to expand a parent whose action
# distribution has entropy below x, i.e. drop OVERCONFIDENT states outright.
# Different mechanism from ENT_BONUS (which re-scores); this removes them.
# WARNING from the measured data: the library/non-library entropy ordering
# FLIPS with depth (steps 0-6 LIB 2.701 vs NON 2.844; steps 45-52 LIB 2.653 vs
# NON 2.089), so a fixed floor culls LIBRARY parents early and non-library ones
# late. Net effect unknown -- measure, do not argue.
# The top-scoring parent is always expanded regardless, so the beam can never
# be emptied by the cut.
_ENT_CUT = float(os.environ.get('SAILIR_ENT_CUT', '0') or 0)
_LINEAGE_TABU = os.environ.get('SAILIR_LINEAGE_TABU') == '1'
_BT_WHO = {}      # (expr_fp, target, action) -> set of (path_len, path_hash)

_SOL_TABU = os.environ.get('SAILIR_SOL_TABU') == '1'
_SOL_FP = {}      # (expr_fp, target, op, delta) -> set of substitution hashes
_SOL_WHY = ['-', 0]   # diagnostic: why the truth action was kept/banned
_SOL_WHO = {}         # same key -> list of writer paths

_MW2_PRICE = float(os.environ.get('SAILIR_MW2_PRICE', '0') or 0)
_SCORE_MODE = os.environ.get('SAILIR_SCORE', 'local')


def _truth_trace_load():
    """Read the truth walk's per-step expression fingerprints, once."""
    global _truth_fp, _trace_fh
    if not _TRUTH_TRACE:
        return
    import json as _json
    with open(_TRUTH_TRACE) as fh:
        for line in fh:
            d = _json.loads(line)
            op, dl = d['chosen_action']
            _truth_actions.append(
                (tuple(d['target']), int(op), tuple(dl)))
    _trace_fh = open(_TRACE_OUT, 'w')
    print(f'[truth-trace] loaded {len(_truth_actions)} truth actions from '
          f'{_TRUTH_TRACE} -> {_TRACE_OUT}', flush=True)


def _truth_enum_check(where, st, target, valid):
    """Is the truth's NEXT action even in the beam's enumerated set?

    Everything else is downstream of this. If the action the truth walk takes
    from this exact state is not in `valid`, then no beam width, no K, no sort
    key and no scoring window can ever follow the truth path -- the beam and the
    recorder simply disagree about what is legal, and comparing them measures
    nothing. Records, for a parent that is ON the truth path:
      target_match  did the beam pick the same integral to eliminate?
      in_valid      is the truth (op, delta) in the enumerated list?
      idx / n       where, out of how many
    """
    if _trace_fh is None or not _truth_actions:
        return
    n = len(st.path)
    if n >= len(_truth_actions):
        return
    if any(st.path[i] != _truth_actions[i] for i in range(n)):
        return                      # this parent is not on the truth path
    ta = _truth_actions[n]
    import json as _j
    tgt_ok = (tuple(target) == ta[0])
    idx = -1
    if tgt_ok:
        try:
            idx = list(valid).index((ta[1], ta[2]))
        except ValueError:
            idx = -1
    _trace_fh.write(_j.dumps(dict(
        event='ENUM', where=where, depth=n, target_match=bool(tgt_ok),
        beam_target=list(target), truth_target=list(ta[0]),
        in_valid=bool(idx >= 0), idx=idx, n_valid=len(valid))) + chr(10))
    _trace_fh.flush()


def _truth_trace_step(step, parents, candidates, beam, sort_w, sort_p,
                      cand_meta=None):
    """How long does the EXACT truth trajectory survive in the beam?

    Matched on the ACTION PATH, not the expression. A beam state whose path
    equals the truth walk's first j actions has by construction the identical
    expression AND the identical substitution store, because both follow from
    replaying the same actions from the same start. That makes the test exact.

    Two earlier versions were unsound and their numbers are withdrawn:
      * comparing expr against _truth_fp[step] assumed beam/truth lockstep;
      * comparing expr against ALL truth indices produced false matches at a
        high rate, because a state is (expr, subs) and truth state j carries j
        substitution entries -- a 1-action beam state was being scored as
        having "reached" a 19-action truth state on expression alone.
    Recovery is not a meaningful notion under the exact test either: a lineage
    that regains a truth EXPRESSION by another route has a different
    substitution store and is a different state, so it is not the truth
    trajectory. Survival is therefore monotone -- once no state carries the
    truth prefix, the trajectory is lost for good.

    Emits per step: the deepest truth prefix alive in the beam, how many states
    carry it, whether the next truth action was generated as a candidate, and
    whether that child survived selection.
    """
    global _frontier
    if _trace_fh is None:
        return
    import json as _json
    tp = _truth_actions
    if not tp:
        return

    def _depth(st):
        n = len(st.path)
        if n <= len(tp) and all(st.path[i] == tp[i] for i in range(n)):
            return n
        return -1

    depths = [_depth(st) for st in beam]
    alive = [d for d in depths if d >= 0]
    best = max(alive) if alive else -1
    if best > _frontier:
        _frontier = best
    # Candidates and the post-selection beam sit at the SAME depth (both are
    # this step's children), so counting candidates at _frontier+1 can never
    # fire -- that was an off-by-one. Compare the deepest truth prefix among
    # CANDIDATES against the deepest that SURVIVED: cand_depth > best means the
    # truth child existed and was culled, which is the distinction that
    # matters.
    cd = [_depth(c) for c in candidates]
    cand_alive = [d for d in cd if d >= 0]
    cand_best = max(cand_alive) if cand_alive else -1
    # When the truth child exists as a candidate but did NOT survive, record
    # its actual keys against the worst survivor's. Every mu/lambda argument so
    # far used numbers from the discredited expression matcher; these are the
    # real ones.
    _cmp = None
    if cand_best > best:
        _tc = next((c for c in candidates if _depth(c) == cand_best), None)
        if _tc is not None and beam:
            _worst = max(beam, key=lambda x: (x.max_w12, x.n_non_masters,
                                              -x.score))
            from collections import Counter as _Ctr
            _mwd = _Ctr(tuple(c.max_w12[:2]) for c in candidates)
            _better = sum(n for mw, n in _mwd.items()
                          if mw < tuple(_tc.max_w12[:2]))
            # score distribution INSIDE the mw pool that outranks the truth
            # child: decides whether any (w1, w2, prob) weighting could lift it
            # into the beam, or whether the better-mw pool is also well-rated.
            _tmw = tuple(_tc.max_w12[:2])
            _btr = sorted(c.score for c in candidates
                          if tuple(c.max_w12[:2]) < _tmw)
            import math as _m
            _same = sorted((c.score for c in candidates
                            if tuple(c.max_w12[:2]) == _tmw), reverse=True)
            _same_rank = sum(1 for x in _same if x > _tc.score)
            _cmp = dict(
                same_mw_n=len(_same), same_mw_rank_by_prob=_same_rank,
                same_mw_score_q=[round(_same[int(q*(len(_same)-1))], 2)
                                 for q in (0, .25, .5, .75, 1)] if _same else [],
                better_mw_n=len(_btr),
                better_mw_p_above_truth=sum(1 for x in _btr
                                            if x > _tc.score),
                better_mw_score_q=[round(_btr[int(q*(len(_btr)-1))], 2)
                                   for q in (0, .25, .5, .75, 1)] if _btr else [],
                cand_mw_hist=sorted((list(k), v) for k, v in _mwd.items())[:6],
                n_cand_better_mw=_better, n_cand=len(candidates),
                truth_mw=list(_tc.max_w12), truth_nm=int(_tc.n_non_masters),
                truth_score=float(_tc.score),
                cut_mw=list(_worst.max_w12), cut_nm=int(_worst.n_non_masters),
                cut_score=float(_worst.score))
    if _LIB is not None:
        _bl = [b for b in beam if _in_library(b)]
        _cl = [c for c in candidates if _in_library(c)]
        # WHY THE CULL PREFERS NON-LIBRARY STATES. Compare the two candidate
        # populations on the exact quantities the sort keys read: score (log
        # prob), max_w12 and n_non_masters. If library candidates are plentiful
        # (they are: 65-99 of ~600 right up to the death) but lose slots, the
        # answer is in the gap between these distributions.
        def _q(v):
            v = sorted(v)
            return [round(v[int(f * (len(v) - 1))], 3)
                    for f in (0, .25, .5, .75, 1)] if v else []
        _nl = [c for c in candidates if not _in_library(c)]
        _rec2 = dict(step=step, n_beam=len(beam), n_lib_beam=len(_bl),
                     n_cand=len(candidates), n_lib_cand=len(_cl),
                     lib_depth=max((len(b.path) for b in _bl), default=-1),
                     best_lib_nm=min((b.n_non_masters for b in _bl),
                                     default=-1),
                     best_lib_mw=(list(min(_bl, key=lambda b: b.max_w12).max_w12)
                                  if _bl else None),
                     beam_best_nm=min((b.n_non_masters for b in beam),
                                      default=-1),
                     lib_score_q=_q([c.score for c in _cl]),
                     non_score_q=_q([c.score for c in _nl]),
                     lib_nm_q=_q([float(c.n_non_masters) for c in _cl]),
                     non_nm_q=_q([float(c.n_non_masters) for c in _nl]),
                     lib_mw_hist=sorted(
                         (list(k), v) for k, v in
                         __import__('collections').Counter(
                             tuple(c.max_w12) for c in _cl).items())[:5],
                     non_mw_hist=sorted(
                         (list(k), v) for k, v in
                         __import__('collections').Counter(
                             tuple(c.max_w12) for c in _nl).items())[:5],
                     # where the surviving beam's worst member sits -- the bar
                     # every library candidate had to clear
                     cut_score=(min(b.score for b in beam) if beam else None),
                     cut_nm=(max(b.n_non_masters for b in beam) if beam else None),
                     lib_parents=list(_LIB_PARENT),
                     all_parents=list(_ALL_PARENT))
        del _LIB_PARENT[:]
        del _ALL_PARENT[:]
        _trace_fh.write(_json.dumps(_rec2) + chr(10))
        _trace_fh.flush()
    # TRUTH CANDIDATE vs THE BEAM THAT DISPLACED IT.
    # Fires whenever the truth child exists as a CANDIDATE but is not in the
    # post-selection beam -- the case `cmp` was meant for but misses, because
    # its trigger (cand_best > best) is false when the beam still holds a
    # SHALLOWER truth state. Records the actual sort keys on both sides so the
    # question "did it lose on weight or on probability" is answered with
    # numbers instead of inference.
    if _trace_fh is not None and cand_alive:
        _tcand = next((c for c in candidates if _depth(c) == cand_best), None)
        _in_beam = any(_depth(b) == cand_best for b in beam)
        if _tcand is not None and not _in_beam and beam:
            import json as _j3
            # FULL keys, never truncated. These fields used to log
            # max_w12[:2] / total_w12[:2], written when those really were
            # 2-tuples. Under SAILIR_BEAM_TOTAL=1 they are the full ordering
            # tuple (r, s, |a|), and slicing dropped the |a| component from BOTH
            # the printout AND the comparisons -- which made the key look
            # degenerate ("every beam state tied") when it was only the readout
            # that was blind. The sort was always using the full key.
            _mw = lambda b: tuple(b.max_w12)
            _tw = lambda b: tuple(b.total_w12)
            _bw = sorted(beam, key=_mw)
            _bt = sorted(beam, key=_tw)
            _trace_fh.write(_j3.dumps(dict(
                event='TRUTH_CAND_DISPLACED', step=step, depth=cand_best,
                beam_total_on=bool(_BEAM_TOTAL),
                truth_max_w12=_mw(_tcand),
                truth_total_w12=_tw(_tcand),
                truth_nm=int(_tcand.n_non_masters),
                truth_score=float(_tcand.score),
                beam_max_w12_best=_mw(_bw[0]),
                beam_max_w12_worst=_mw(_bw[-1]),
                beam_total_w12_best=_tw(_bt[0]),
                beam_total_w12_worst=_tw(_bt[-1]),
                n_distinct_maxw=len({_mw(b) for b in beam}),
                truth_maxw_rank=int(sum(1 for b in beam
                                        if _mw(b) < _mw(_tcand))) + 1,
                beam_score_best=float(max(b.score for b in beam)),
                beam_score_worst=float(min(b.score for b in beam)),
                n_beam_better_maxw=int(sum(
                    1 for b in beam if _mw(b) < _mw(_tcand))),
                n_beam_better_totalw=int(sum(
                    1 for b in beam if _tw(b) < _tw(_tcand))),
                n_beam_better_score=int(sum(
                    1 for b in beam if b.score > _tcand.score)),
                n_beam=len(beam))) + chr(10))
            _trace_fh.flush()

    rec = dict(step=step, truth_depth=best, n_on_truth=len(alive),
               cand_depth=cand_best, n_cand_on_truth=len(cand_alive),
               culled=bool(cand_best > best), cmp=_cmp,
               deepest_ever=_frontier,
               n_cand=len(candidates), n_beam=len(beam))
    _trace_fh.write(_json.dumps(rec) + chr(10))
    _trace_fh.flush()

if _SCORE_MODE not in ('local', 'cumulative', 'decay'):
    raise SystemExit(f"unknown SAILIR_SCORE={_SCORE_MODE!r} "
                     f"(local | cumulative | decay)")
# 'decay' = cumulative WITH FORGETTING:  score_t = gamma*score_{t-1} + log p_t
# A geometric window instead of a hard one -- no per-state deque, still one
# float. gamma=0 reproduces 'local' exactly and gamma=1 reproduces
# 'cumulative', so the three modes are one continuous knob.
#
# Why forgetting may beat both endpoints:
#   * cumulative never revises stale evidence -- a lineage is judged forever on
#     a step-30 probability that described a state 800 steps ago;
#   * cumulative also COMPRESSES: by step 800 every survivor sums to ~-800 and
#     the differences that decide the ranking shrink toward the tiebreakers;
#   * local keeps full scale but has one-step memory, so 800 good decisions
#     count for nothing.
# The decayed sum is bounded (geometric), so it keeps the model's signal on a
# fixed scale against the nm penalty for the whole run.
_SCORE_DECAY = float(os.environ.get('SAILIR_SCORE_DECAY', '0.9'))
if not 0.0 <= _SCORE_DECAY <= 1.0:
    raise SystemExit(f"SAILIR_SCORE_DECAY must be in [0,1], got {_SCORE_DECAY}")

_ACTION_SCORE = os.environ.get('SAILIR_ACTION_SCORE', 'model')
if _ACTION_SCORE not in ('model', 'sumseed', 'random'):
    raise SystemExit(f"unknown SAILIR_ACTION_SCORE={_ACTION_SCORE!r} "
                     f"(model | sumseed | random)")


def _model_free_row(target, valid, n_v):
    """Per-action scores with the model removed. Higher is better, matching
    the model's probabilities, so every downstream consumer (top-K, tabu,
    sort_prob) works unchanged."""
    if _ACTION_SCORE == 'sumseed':
        return np.array(
            [-float(sum(abs(t + d) for t, d in zip(target, valid[i][1])))
             for i in range(n_v)], dtype=np.float64)
    import zlib
    seed = zlib.crc32(repr((tuple(target), n_v)).encode()) & 0xffffffff
    return np.random.default_rng(seed).random(n_v)

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

# EQACT inference: resolved equations from the LIVE store. Imported lazily
# inside the model call rather than at module import, so a nosubs run never
# pays for it and never depends on the training package.
_eqi = None


def _eqi_mod():
    global _eqi
    if _eqi is None:
        import eqact_infer as _m
        _eqi = _m
    return _eqi
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
    if _STRIP_TOTAL and _START_TOTAL_KEY is not None:
        # Strictly BELOW the start in the total ordering => passenger.
        # Built from the w we already have instead of calling _target_key(),
        # which would recompute weight() (and _sector_mask()) on every term of
        # every state. When the sector test above ran, every surviving integral
        # shares the start's rank prefix, so comparing the (r,s,|abs|) tail is
        # equivalent to comparing the full key.
        # tkey-order tail, computed directly (weight() deviates)
        # tkey order (smaller = higher) computed DIRECTLY from the
        # indices -- never by re-signing weight()'s components.
        tail = (-sum(x for x in integral if x > 0),
                -sum(-x for x in integral if x < 0),
                tuple(abs(x) for x in integral))
        if not _SECTOR_RANK:
            if tail > _START_TOTAL_KEY:
                return False
        elif _START_SECTOR is not None:
            if tail > _START_TOTAL_KEY[1:]:
                return False
        elif _target_key(integral) > _START_TOTAL_KEY:
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

# --- TOTAL-ORDER PASSENGER STRIP -------------------------------------------
# The strip decides what stays in the expression. It used to keep everything at
# (w1,w2) >= the start's, which retains integrals at the SAME (r,s) but larger
# |abs| -- those are strictly BELOW the start in the total ordering, so under
# the total-order success criterion they are passengers: they can never be
# targeted (targets are min(nm, key=_target_key), and anything below the start
# loses to the start itself) and they cannot affect whether the step succeeds.
# Carrying them costs memory and model attention for nothing. The truth
# recorder already strips exactly this way (_active: same sector and
# tkey(k) <= tkey(T)), so turning it on also removes a train/inference mismatch.
#
# DEFAULT FOLLOWS _SUCCESS_TOTAL, and the mixed setting is refused, because
# stripping by total order while success is still the (w1,w2) bucket test would
# declare success EARLY: the same-(r,s)-larger-|abs| terms would be dropped from
# the expression rather than eliminated, so n_non_masters could reach 0 with
# real work outstanding. Both workers force SAILIR_SUCCESS_TOTAL=1 before
# importing this module, so worker-driven runs get the strip on by default.
_STRIP_TOTAL = os.environ.get(
    'SAILIR_STRIP_TOTAL', '1' if _SUCCESS_TOTAL else '0') == '1'
if _STRIP_TOTAL and not _SUCCESS_TOTAL:
    raise SystemExit(
        'SAILIR_STRIP_TOTAL=1 requires SAILIR_SUCCESS_TOTAL=1: stripping by '
        'the total order while success is the (w1,w2) bucket test would '
        'declare success before the same-(r,s)/larger-|abs| terms are '
        'eliminated.')


def _target_key(i):
    """Total-ordering key matching the training-data target selection:
    (-r, -s, |abs|-tuple). Smaller = higher in the total ordering.

    DESIGN DECISION 2026-07-10 (see reduction/ORDERING.md): the adopted order is
    SECTOR RANK first, then (r,s), then |abs|. Default = LEGACY order;
    SAILIR_SECTOR_RANK=1 switches to the adopted order (rank prefix). The flag
    must be set identically for orchestrator + workers + symmetry_route +
    canonical_rep of a run (the shared order keeps the substitution cache
    acyclic)."""
    # tkey's order is SMALLER = higher and DELIBERATELY deviates from
    # weight() (which is larger = higher, with |abs| already negated).
    # Compute it directly instead of deriving it from weight -- deriving it
    # would need a per-component sign fix, the exact mismatch we removed.
    base = (-sum(x for x in i if x > 0), -sum(-x for x in i if x < 0),
            tuple(abs(x) for x in i))
    if not _SECTOR_RANK:
        return base
    return (-_RANK_IDX[_sector_mask(i)],) + base


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
    if _SOL_TABU:
        # free: sol is computed here anyway to apply the action
        _SOL_FP.setdefault(
            (frozenset(state.expr.items()), tuple(target), ibp_op,
             tuple(delta)), set()).add(hash(frozenset(sol.items())))
        # who wrote it: the writing state's path. Lets us ask whether a ban
        # came from this lineage's own ancestor (a genuine loop) or from an
        # unrelated lineage that merely passed through the same expression.
        _SOL_WHO.setdefault(
            (frozenset(state.expr.items()), tuple(target), ibp_op,
             tuple(delta)), []).append(tuple(state.path))
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
    # score = LOCAL log-prob of the action just taken, or CUMULATIVE path
    # log-prob with SAILIR_SCORE=cumulative.
    #
    # The original justification for local-only was that cumulative scoring
    # "biases against long lineages". That does not apply here: the beam
    # advances in lockstep, so EVERY state in the beam at step k has taken
    # exactly k actions -- verified, 0/100 runs had non-empty restart_offsets,
    # so paths are unbroken chains. Comparing cumulative scores within a step
    # compares sums of equally many terms, and length-normalising would divide
    # every candidate by the same k, leaving the order unchanged.
    #
    # What local scoring actually does is give the ranking ONE-STEP MEMORY: at
    # step 4,881 the 4,880 preceding decisions count for nothing, and a state
    # reached by a poor route outranks a well-reached one whenever its last
    # move happened to look easy. The remaining honest argument for local is
    # that cumulative scoring holds one early improbable step against a lineage
    # forever -- which is either a bug or correct bookkeeping depending on
    # whether the model's early probabilities mean anything.
    _lp = math.log(action_prob + 1e-10)
    if _SCORE_MODE == 'cumulative':
        new_score = state.score + _lp
    elif _SCORE_MODE == 'decay':
        new_score = _SCORE_DECAY * state.score + _lp
    else:
        new_score = _lp
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
        lane=state.lane,      # DUAL: a child stays in its parent's sub-beam
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
# visits.
# (An older NOTE here claimed a train/inference target-ordering mismatch -- that
# the beam targeted ALL (r,s)-tied integrals while training data targets ONE by
# full order. STALE, removed 2026-08-23: every target site is
# `tied = [min(nm, key=_target_key)]` (4 sites, each marked "v8: SINGLE
# full-order target"), and _target_key IS the training-data order (-r, -s,
# |abs|). The `tied` name is a leftover from the old behaviour.)
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


# ===========================================================================
# v9: UPSTREAM FASTMAXW CULL -- the action space the model actually sees.
# ===========================================================================
# Replaces the positional `first900` cap (valid[:max_actions], i.e. the OLDEST
# actions in enumeration order, with no notion of quality) with a ranked cut.
#
# For each enumerated candidate (op, delta), with seed = target + delta:
#     raw_terms = {seed + shift for shift in shifts[op]}     integer arithmetic
#     admissible  iff  target in raw_terms                        (direct)
#                 or   target in sol(K) for a store key K in raw_terms (indirect)
#     maxw_bound  = max over K in raw_terms of
#                       store_maxw[K]  if K is a store key
#                       weight(K)[:2]  otherwise
#
# resolved_subs is fully back-substituted, so one pass is final -- no recursion.
# No row is materialised, no eval_coeff, no substitution: ~30 dict lookups and
# integer maxes per candidate, with store_maxw built ONCE per step.
#
# Both quantities ignore coefficients, so this OVER-ADMITS (a raw term whose
# coefficient is zero, or a target that cancels, still counts) and the bound is
# an UPPER bound on the true resolved maxw. Measured on real states: the bound
# equals the exact resolved maxw 99.9% of the time, is never below it, and
# selects an identical top-500. Over-admission is harmless downstream -- an
# action that cannot act simply fails when applied.
#
# Validated end to end (2026-08-24, gravity3L, width-1 truth-closure replay):
# solves 57/60 vs 58/60 for the exact resolved-maxw ranking, at 0.434 s/step
# vs 1.960 s. The fine-tuning corpus (1844 targets / 84,963 samples) was
# generated with exactly this cull, so a model trained on it sees the same
# action space here that it saw in training.
_V9_CULL = os.environ.get('SAILIR_V9_CULL', '1') != '0'


# ---- CLOSURE-COVERAGE PROBE (DAgger feasibility) --------------------------
# SAILIR_CLOSURE_PROBE=<closure.json> loads a target's truth closure and, at
# every beam state, counts how many of that state's LEGAL actions are UNUSED
# CLOSURE ROWS -- exactly the truthminnew availability condition
# (beam_search_truthcull.py: "keep only UNUSED CLOSURE rows -> take the best").
#
# The question it answers: when the model deviates from the truth path, can the
# expert still label the state? A task with zero available rows is one where
# truthminnew has NO action, so DAgger could not produce a label there.
# 'used' is per-state and reconstructed from that state's own path, since each
# beam state consumed a different set of rows.
# ---- DAgger ROW EMITTER ---------------------------------------------------
# SAILIR_DAGGER_OUT=<rows.jsonl>  emit training rows at states the MODEL
#   actually visits, rather than at states a truthminnew walk visits. This is
#   the DAgger step: the expert labels the learner's own state distribution.
#
# The label rule is IDENTICAL to the production recorder's. From
# data-gen/preprocess_to_tensors.py: "valid_label_idxs: EVERY unused closure
# row that solves this target, as indices into valid_actions. All of them are
# correct actions, so 'label' is one arbitrary pick from this set." So the
# expert here is exactly the corpus's expert, just queried off-path -- which is
# sound because a closure is a SPANNING SET for the elimination, not a path
# (measured: 0 exhaustions over ~1,240 beam states on 4 targets).
#
# SAILIR_DAGGER_MODE:
#   errors (default) -- emit ONLY where the model's top-1 is NOT a correct
#     action. truthminnew is WEAKER than the model where the model works (96-
#     step truth walks vs the beam's 24), so imitating it everywhere would cap
#     the learner at the worse policy. Labelling only the model's mistakes
#     fixes recovery without overwriting good routing.
#   all -- emit every state (classic DAgger aggregation).
# SAILIR_DAGGER_DEADEND=1 -- also emit states where NO unused closure row is
#   legal, with an EMPTY valid_label_idxs. Those are evidenced dead ends (the
#   expert itself has no move), not merely unfamiliar states, so an all-zeros
#   BCE target there is a fact rather than an assumption.
_DAGGER_OUT = os.environ.get('SAILIR_DAGGER_OUT')
_DAGGER_MODE = os.environ.get('SAILIR_DAGGER_MODE', 'errors')
_DAGGER_DEADEND = os.environ.get('SAILIR_DAGGER_DEADEND') == '1'
_DAGGER_FH = None
_DAGGER_STATS = {'emitted': 0, 'skipped_model_right': 0, 'deadend': 0, 'seen': 0}

# SAILIR_DAGGER_MISSING=<targets.txt> -- record every TARGET whose closure ran
# out (no unused closure row is legal at a state the model reached). This is
# the work list for closure REGENERATION.
#
# Regeneration is only meaningful at a HIGHER RUNG. truth_engine.worker_replay
# walks the ladder [(dr,ds), (dr+1,ds), ... ] and BREAKS at the first rung that
# solves T, so for fixed (dr, ds, SAILIR_SEED_BUDGET) it is deterministic:
# re-running it reproduces the same closure row for row, and would add nothing.
# A larger seed box is a strict superset of equations, so escalating dr/ds (and
# raising the seed budget, which is what makes the deeper rungs reachable at
# all rather than `continue`-skipped) is what actually produces new rows.
# reduction/run_closure_regen.sh drives that.
_DAGGER_MISSING = os.environ.get('SAILIR_DAGGER_MISSING')
_DAGGER_MISSING_SET = set()
if _DAGGER_OUT:
    import atexit as _dax
    def _dagger_summary():
        st = _DAGGER_STATS
        if st['seen']:
            print(f"  [DAGGER] seen={st['seen']} emitted={st['emitted']} "
                  f"skipped_model_already_right={st['skipped_model_right']} "
                  f"deadend={st['deadend']} "
                  f"model_error_rate={(st['emitted']+st['deadend'])/st['seen']:.3f}",
                  flush=True)
        if _DAGGER_MISSING and _DAGGER_MISSING_SET:
            # append + dedupe on read; several collector runs share one file
            with open(_DAGGER_MISSING, 'a') as _mf:
                for _t in sorted(_DAGGER_MISSING_SET):
                    _mf.write(','.join(str(int(x)) for x in _t) + '\n')
            print(f"  [DAGGER] {len(_DAGGER_MISSING_SET)} exhausted target(s) "
                  f"-> {_DAGGER_MISSING}", flush=True)
    _dax.register(_dagger_summary)

# SAILIR_CLOSURE_PROBE accepts a ':'-separated list of closure JSONs and/or
# DIRECTORIES of them; every entry's rows are UNIONed. A single file path (the
# original form) still behaves exactly as before.
#
# The union matters for regeneration. One closure is the dependency closure of
# its own target's rule, so the campaign target's closure normally covers the
# whole reduction. When the model goes off-path it can reach an intermediate
# target that closure does not span; the fix is a closure built FOR THAT
# TARGET, which arrives as an extra file. Rows are (op, seed) with seed
# absolute, so unioning across targets is well defined -- the emitter's
# membership test is over absolute seeds too.
_CLOSURE_PROBE = os.environ.get('SAILIR_CLOSURE_PROBE')
_CLOSURE_PROBE_SET = None
if _CLOSURE_PROBE:
    import glob as _clg
    import json as _clj
    _cl_files = []
    for _ent in _CLOSURE_PROBE.split(':'):
        if not _ent:
            continue
        if os.path.isdir(_ent):
            _cl_files.extend(sorted(_clg.glob(os.path.join(_ent, '*.json'))))
        else:
            _cl_files.append(_ent)
    _CLOSURE_PROBE_SET = set()
    _cl_ok = 0
    for _f in _cl_files:
        try:
            with open(_f) as _clf:
                _CLOSURE_PROBE_SET.update((int(_o), tuple(_sd))
                                          for _o, _sd in _clj.load(_clf)['closure'])
            _cl_ok += 1
        except Exception as _e:
            print(f'  [CLOSUREPROBE] SKIP {_f}: {type(_e).__name__}: {_e}',
                  flush=True)
    print(f'  [CLOSUREPROBE] loaded {len(_CLOSURE_PROBE_SET)} closure rows '
          f'from {_cl_ok}/{len(_cl_files)} file(s)', flush=True)


# Cache keyed on the STORE OBJECT, not the step. _v9_cull runs once per
# (state, target) TASK, and _rs_as_dict materialises every PackedEq in the store
# into a Python dict -- so at beam 40 the uncached version rebuilt the whole
# substitution store ~38 times per step. Beam states share their store object
# until a substitution is added, and apply_action_v5 builds a NEW dict for the
# child, so identity is a sound key: a mutated store is a different object.
# Bounded by construction -- only the current beam's stores are reachable, and
# the entry dies with the state.
_V9_SMW_CACHE = {}


# ── weight, without the waste ──────────────────────────────────────────────
# ibp_env.weight() makes THREE passes over the indices and allocates a 15-tuple
# of absolute values on every call. The cull uses only (w1, w2) and discards
# that tuple immediately. cProfile over a beam-40 run: 4,332,313 calls, ~59s
# cumulative out of ~110s -- the whole of the step's unexplained "residual".
#
# _w12 does ONE pass, returns the two scalars, allocates nothing, and memoises
# on the integral (a pure function of it, and the same integrals recur heavily
# across candidates, beam slots and steps). The cache is bounded: integrals
# seen once at depth are unlikely to return, so a hard cap with a clear beats
# unbounded growth.
import operator as _operator
_ADD = _operator.add
_W12_CACHE = {}
_W12_CAP = 2_000_000


def _w12(i):
    """(sum of positive indices, -sum of negative indices). Memoised."""
    v = _W12_CACHE.get(i)
    if v is not None:
        return v
    a = b = 0
    for x in i:
        if x > 0:
            a += x
        elif x < 0:
            b -= x
    v = (a, b)
    if len(_W12_CACHE) >= _W12_CAP:
        _W12_CACHE.clear()
    _W12_CACHE[i] = v
    return v


def _v9_store_maxw(resolved_subs):
    """({sub_int: maxw of its solution}, dict-view of the store).

    One scalar per substitution. The store is back-substituted, so exact."""
    key = id(resolved_subs)
    hit = _V9_SMW_CACHE.get(key)
    if hit is not None and hit[0] is resolved_subs:
        return hit[1], hit[2]
    d = _rs_as_dict(resolved_subs)
    smw = {k: max((_w12(i) for i in v), default=(0, 0))
           for k, v in d.items()}
    if len(_V9_SMW_CACHE) > 4096:          # stale ids from dead states
        _V9_SMW_CACHE.clear()
    _V9_SMW_CACHE[key] = (resolved_subs, smw, d)
    return smw, d


# --- ORACLE / VERIFY: the stringent alignment test -------------------------
# SAILIR_V9_ORACLE=<ftdata .jsonl>  replay the truth actions recorded for this
#   integral INSTEAD of consulting the model, at beam_width 1. Two invariants
#   are then checked at every step, and any violation is FATAL rather than
#   warned about, because a silent mismatch is exactly what makes a training
#   corpus quietly useless:
#     (1) the oracle's action is present in v9's culled action space
#     (2) v9's culled action space EQUALS the training sample's valid_actions,
#         as an ordered list -- same members, same rank order, so a label index
#         means the same thing at inference as it did in training.
_V9_ORACLE_PATH = os.environ.get('SAILIR_V9_ORACLE')
_V9_ORACLE = None
if _V9_ORACLE_PATH:
    import json as _oj
    _V9_ORACLE = []
    with open(_V9_ORACLE_PATH) as _of:
        for _line in _of:
            if _line.strip():
                _V9_ORACLE.append(_oj.loads(_line))
    print(f'[v9-oracle] loaded {len(_V9_ORACLE)} recorded steps from '
          f'{_V9_ORACLE_PATH}', flush=True)
_V9_ORACLE_STRICT = os.environ.get('SAILIR_V9_ORACLE_STRICT', '1') != '0'
_V9_ORACLE_STATS = {'steps': 0, 'space_match': 0, 'action_in_space': 0,
                    'space_mismatch': [], 'action_missing': []}


# The recorded action sequence, for ON-PATH matching at beam_width > 1.
# A state whose path equals the truth walk's first j actions has BY
# CONSTRUCTION the identical expression and substitution store (both follow
# from replaying the same actions from the same start), so the comparison is
# exact for that state. Matching on the step INDEX instead would compare a
# divergent state's space against a record from a different state -- which is
# meaningless and would report spurious mismatches.
_V9_TRUTH_ACTIONS = ([(r['chosen_action'][0], tuple(r['chosen_action'][1]))
                      for r in _V9_ORACLE] if _V9_ORACLE else [])


def _v9_on_truth_path(state):
    """Is this beam state still exactly on the recorded trajectory?"""
    n = len(state.path)
    if n > len(_V9_TRUTH_ACTIONS):
        return False
    for i in range(n):
        _t, _op, _d = state.path[i]
        if (_op, tuple(_d)) != _V9_TRUTH_ACTIONS[i]:
            return False
    return True


_V9_ORACLE_RANK = os.environ.get('SAILIR_V9_ORACLE_RANK') not in (None, '', '0')
_V9_RANK_PENDING = {}          # task index -> (depth, action, cull rank, space)
_V9_RANKS = []                 # (depth, cull_rank, model_rank, space, p, p_top)


def _v9_oracle_check(step, target, kept):
    """Compare v9's culled space against the recorded one. Returns the oracle
    action for this step, or None if there is no record for it."""
    if _V9_ORACLE is None or step >= len(_V9_ORACLE):
        return None
    rec = _V9_ORACLE[step]
    st = _V9_ORACLE_STATS
    st['steps'] += 1
    rec_t = tuple(rec['target'])
    if rec_t != tuple(target):
        msg = (f'[v9-oracle] step {step}: TARGET MISMATCH '
               f'recorded={list(rec_t)} v9={list(target)}')
        if _V9_ORACLE_STRICT:
            raise SystemExit(msg)
        print(msg, flush=True)
        return None
    rec_space = [(a[0], tuple(a[1])) for a in rec['valid_actions']]
    v9_space = [(a[0], tuple(a[1])) for a in kept]
    if rec_space == v9_space:
        st['space_match'] += 1
    else:
        rs, vs = set(rec_space), set(v9_space)
        st['space_mismatch'].append(
            (step, len(rec_space), len(v9_space), len(rs - vs), len(vs - rs),
             rec_space == sorted(v9_space, key=v9_space.index)))
        msg = (f'[v9-oracle] step {step}: ACTION SPACE MISMATCH '
               f'recorded={len(rec_space)} v9={len(v9_space)} '
               f'rec_only={len(rs - vs)} v9_only={len(vs - rs)} '
               f'same_set={rs == vs}')
        if _V9_ORACLE_STRICT:
            raise SystemExit(msg)
        print(msg, flush=True)
    act = (rec['chosen_action'][0], tuple(rec['chosen_action'][1]))
    if act in set(v9_space):
        st['action_in_space'] += 1
    else:
        st['action_missing'].append((step, act))
        msg = (f'[v9-oracle] step {step}: ORACLE ACTION NOT IN v9 SPACE '
               f'op{act[0]} delta={list(act[1])} (space={len(v9_space)})')
        if _V9_ORACLE_STRICT:
            raise SystemExit(msg)
        print(msg, flush=True)
    return act


def _v9_oracle_report():
    st = _V9_ORACLE_STATS
    if not st['steps']:
        return
    print(f'[v9-oracle] steps checked          : {st["steps"]}', flush=True)
    print(f'[v9-oracle] action space MATCHED   : {st["space_match"]}/{st["steps"]}')
    print(f'[v9-oracle] oracle action in space : {st["action_in_space"]}/{st["steps"]}')
    print(f'[v9-oracle] deepest ON-PATH state  : step '
          f'{st.get("deepest_on_path", 0)}', flush=True)
    ok = (st['space_match'] == st['steps'] == st['action_in_space'])
    print(f'[v9-oracle] VERDICT: {"PASS" if ok else "FAIL"}', flush=True)


# ── Store-only enumeration (ported verbatim from beam_search_truthcull.py) ──
# Replaces the cu-based enumeration: cu is never built, unpacked, scanned or
# extended. A move is legal exactly when its equation contains a term that is
# the target, or a solved integral whose solution contains the target -- both
# readable from resolved_subs plus the identity templates, with no equation
# materialised. Two exact tests keep the candidate set tight:
#   vanishing coefficient  most coefficients are one index times a constant,
#                          so the term is absent iff seed[k] == 0
#   banned integral        100% of these trace to the identity's own raw
#                          terms, never to a substitution, so the test is
#                          exact rather than a bound
# Measured (853-step target, beam 1): 38:32 -> 20:44 wall, 1507MB -> 682MB
# peak, identical path. Verified missing=0 against real enumeration at every
# step of two targets.
_V9_UPENUM = os.environ.get('SAILIR_V9_UPENUM') not in (None, '', '0')
_V9_SEC_CULLED = 0


_TC_AVAR = re.compile(r'a(\d+)')
_TC_VANISH = None
_TC_VEC = None
_TC_LIN = None


def _tc_single_index(coeff):
    """Index k when the coefficient is a single PRODUCT holding exactly one
    a_k ('-a3', '2*a2', '-a12*y', '1/2*a13'), else None. Such a coefficient is
    zero exactly when seed[k] == 0; the constant/kinematic factors are non-zero
    so they never move that condition. A top-level sum has no single condition:
    strip one leading sign, then a remaining +/- means sum.

    Measured on gravity3L: 320 of 332 (op, shift) pairs qualify."""
    s = coeff.strip().replace(' ', '')
    if s[:1] in '+-':
        s = s[1:]
    if '+' in s or '-' in s:
        return None
    idx = _TC_AVAR.findall(s)
    return int(idx[0]) if len(idx) == 1 else None


def _tc_vanish_table(env):
    """[(shift, k_or_None, unit, coeff_str)] per operator, aligned with the
    templates. Built once -- it depends only on the topology.

    `unit` is the coefficient evaluated with index k set to 1. Single-index
    coefficients are DEGREE ONE in that index, so the coefficient at any seed
    is just unit * seed[k] mod PRIME -- no eval, ever, for 96% of offsets.
    That linearity is ASSERTED against eval_coeff on real seeds below rather
    than trusted: if it failed, the residue filter would silently drop good
    moves. Entries we cannot read this way carry unit=None and fall back to
    eval_coeff.
    """
    global _TC_VANISH
    if _TC_VANISH is not None:
        return _TC_VANISH
    n_idx = ibp_env.N_INDICES

    def entry(sh, c):
        k = _tc_single_index(c)
        if k is None:
            return (sh, None, None, c)
        probe = [0] * n_idx
        probe[k] = 1
        unit = ibp_env.eval_coeff(c, tuple(probe))
        return (sh, k, unit, c)

    tbl = {}
    n_ibp = len(env.ibp_t)
    for op, terms in env.ibp_t.items():
        tbl[op] = [entry(sh, c) for sh, c in terms]
    for li, terms in env.li_t.items():
        tbl[n_ibp + li] = [entry(sh, c) for sh, c in terms]

    # Verify linearity on a spread of index values before anything relies on it.
    P = ibp_env.PRIME
    bad = 0
    for op, terms in tbl.items():
        for sh, k, unit, c in terms:
            if k is None:
                continue
            for v in (-3, -1, 2, 5, 11):
                probe = [1] * n_idx
                probe[k] = v
                if ibp_env.eval_coeff(c, tuple(probe)) != (unit * v) % P:
                    bad += 1
                    break
    if bad:
        raise RuntimeError(
            f'_tc_vanish_table: {bad} coefficient(s) are NOT linear in their '
            'single index -- the residue filter would be wrong. Refusing to '
            'run.')
    _TC_VANISH = tbl
    return tbl


def _tc_sol_contains(sol, target, tid):
    """Is `target` a term of this resolved solution? PackedEq keeps ids sorted
    and unique, so membership is one binary search; dict values fall back to
    plain membership."""
    ids = getattr(sol, 'ids', None)
    if ids is None:
        return target in sol
    n = ids.shape[0]
    if n == 0:
        return False
    j = int(np.searchsorted(ids, tid))
    return j < n and int(ids[j]) == tid


def _tc_sol_coeff(sol, target, tid):
    """The target's coefficient inside a resolved solution, or 0 if absent."""
    ids = getattr(sol, 'ids', None)
    if ids is None:
        return sol.get(target, 0)
    n = ids.shape[0]
    if n == 0:
        return 0
    j = int(np.searchsorted(ids, tid))
    if j < n and int(ids[j]) == tid:
        return int(sol.coeffs[j])
    return 0


def _tc_target_survives(seed, terms, resolved_subs, target, tid, n_idx,
                        rc_row):
    """Does the target survive into the substituted equation with a NON-ZERO
    coefficient? Exact test for the last of the residue -- both "target never
    appears" and "target cancels".

        coeff = raw[target]                        (if the target is a term)
              + sum over raw terms K that are solved integrals of
                    raw[K] * (target's coefficient inside sol(K))

    rc_row holds this operator's raw coefficients AT THIS SEED, computed for
    every survivor in ONE batched matmul by the caller. Doing that matmul per
    candidate instead cost more in numpy call overhead than the arithmetic it
    replaced -- 5,255 tiny (16x15) products per generator call.
    """
    P = ibp_env.PRIME
    total = 0
    for j, (sh, _ki, _u, _c) in enumerate(terms):
        rc = rc_row[j]
        if rc == 0:
            continue                    # coefficient vanishes: term absent
        k = tuple(map(operator.add, seed, sh))
        if k == target:
            total += rc
            continue
        sol = resolved_subs.get(k)
        if sol is not None:
            tc = _tc_sol_coeff(sol, target, tid)
            if tc:
                total += rc * tc
    return total % P != 0


def _tc_vec_tables(env, n_idx):
    """Per-operator numpy views of the templates, built once.

    shifts (m_o, n_idx) int32 term offsets; vidx (m_o,) int32 the index whose
    value zeroes that coefficient, or -1 for a sum.
    """
    global _TC_VEC
    if _TC_VEC is not None:
        return _TC_VEC
    tbl = _tc_vanish_table(env)
    vec = {}
    for op, terms in tbl.items():
        sh = np.array([t[0] for t in terms], dtype=np.int32)
        vi = np.array([-1 if t[1] is None else t[1] for t in terms],
                      dtype=np.int32)
        vec[op] = (sh, vi)
    _TC_VEC = vec
    return vec


def _tc_lin_table(env, n_idx):
    """Every identity coefficient as a LINEAR FORM over the indices.

    Each coefficient is degree one in the indices -- the single-index ones
    obviously, and the sums ('d-2*a2-a12-a9-a6-a3-a13', 'a2-a9', ...) too. So
    a coefficient is a vector: coeff(seed) = c0 + sum_i c_i * seed_i, mod
    PRIME. Extracted once by evaluating at zero and at each unit vector, which
    removes eval_coeff from the hot path -- it was ~3,400 Python eval() calls
    per generator call.

    Also SUBSUMES the single-index vanishing test and is strictly more
    accurate: a sum coefficient that vanishes at this seed is now detected,
    where before it was assumed non-zero.

    Linearity is verified against eval_coeff; a mismatch raises rather than
    silently mis-scoring actions.
    """
    global _TC_LIN
    if _TC_LIN is not None:
        return _TC_LIN
    P = ibp_env.PRIME
    tbl = _tc_vanish_table(env)
    zero = tuple([0] * n_idx)
    lin = {}
    for op, terms in tbl.items():
        M = np.zeros((len(terms), n_idx + 1), dtype=np.int64)
        for j, (_sh, _k, _u, cstr) in enumerate(terms):
            c0 = ibp_env.eval_coeff(cstr, zero) % P
            M[j, n_idx] = c0
            for i in range(n_idx):
                e = [0] * n_idx
                e[i] = 1
                M[j, i] = (ibp_env.eval_coeff(cstr, tuple(e)) - c0) % P
        lin[op] = M
    bad = 0
    for op, terms in tbl.items():
        M = lin[op]
        for t in range(4):
            seed = tuple(((i * 7 + t * 13) % 9) - 4 for i in range(n_idx))
            sv = np.array(seed, dtype=np.int64)
            pred = (M[:, :n_idx] @ sv + M[:, n_idx]) % P
            for j, (_sh, _k, _u, cstr) in enumerate(terms):
                if ibp_env.eval_coeff(cstr, seed) % P != int(pred[j]) % P:
                    bad += 1
    if bad:
        raise RuntimeError(
            f'_tc_lin_table: {bad} coefficient(s) are NOT linear in the '
            'indices -- the residue filter would mis-score actions.')
    _TC_LIN = lin
    return lin


def _tc_upenum_candidates(target, resolved_subs, env, reg, n_idx):
    """Vectorised store-only candidate generation.

    Same three exact tests as the scalar version, but the bulk of the work is
    array arithmetic instead of ~600k interpreted operations per call:

      seeds        T - shift for every T in S and every offset, as one
                   (|S|, m_o, n_idx) block per operator
      vanishing    gather the one index each coefficient depends on and keep
                   the candidates where it is non-zero
      banned       a term leaves the target's family iff some position where
                   the target is non-positive is positive there -- one boolean
                   reduction over the few such positions

    Only the (candidate, term) pairs the family test FLAGS need the store
    lookup and strip check, and those are few, so they stay in Python. The
    surviving-target-coefficient filter likewise runs only on survivors.
    """
    tid = reg.get_id(target)
    S = [target]
    for k, sol in resolved_subs.items():
        if _tc_sol_contains(sol, target, tid):
            S.append(k)
    vec = _tc_vec_tables(env, n_idx)
    tbl = _tc_vanish_table(env)
    lin = _tc_lin_table(env, n_idx)
    outside = np.array([i for i in range(n_idx) if target[i] <= 0],
                       dtype=np.intp)
    thr = ibp_env.get_raw_strip_threshold()
    thr2 = tuple(thr)[:2] if thr is not None else None
    S_arr = np.array(S, dtype=np.int32)                   # (nS, n_idx)
    tgt = np.array(target, dtype=np.int32)

    kept_ops = []
    kept_deltas = []
    n_bad = 0
    for op, (sh, vi) in vec.items():
        # seeds: (nS, m, n_idx)
        seeds = S_arr[:, None, :] - sh[None, :, :]
        m = sh.shape[0]
        # vanishing coefficient: seed[vidx] == 0 -> term absent -> not a
        # candidate from THIS offset
        gi = np.where(vi >= 0, vi, 0)
        gathered = np.take_along_axis(
            seeds, np.broadcast_to(gi[None, :, None],
                                   (seeds.shape[0], m, 1)), axis=2)[:, :, 0]
        alive = (vi < 0)[None, :] | (gathered != 0)
        if not alive.any():
            continue
        cand = seeds[alive]                               # (nc, n_idx)
        if cand.shape[0] == 0:
            continue
        # dedup identical seeds (different T can reach the same one)
        cand = np.unique(cand, axis=0)

        # family test: term k = seed + shift; bad iff any OUTSIDE position > 0
        kk = cand[:, None, :] + sh[None, :, :]            # (nc, m, n_idx)
        # a term only counts when its own coefficient does not vanish
        gi2 = np.broadcast_to(gi[None, :, None], (cand.shape[0], m, 1))
        g2 = np.take_along_axis(cand[:, None, :].repeat(m, 1), gi2,
                                axis=2)[:, :, 0]
        term_live = (vi < 0)[None, :] | (g2 != 0)
        if outside.size:
            bad = (kk[:, :, outside] > 0).any(axis=2) & term_live
        else:
            bad = np.zeros((cand.shape[0], m), dtype=bool)

        flagged = np.flatnonzero(bad.any(axis=1))
        ok = np.ones(cand.shape[0], dtype=bool)
        for ci in flagged:                     # few: only flagged candidates
            row = np.flatnonzero(bad[ci])
            for j in row:
                key = tuple(int(x) for x in kk[ci, j])
                if key in resolved_subs:
                    continue                   # replaced by its solution
                if thr2 is not None and _w12(key) < thr2:
                    continue                   # stripped: never generated
                ok[ci] = False
                break
        surv = cand[ok]
        n_bad += int(cand.shape[0] - surv.shape[0])
        if surv.shape[0]:
            # Keep survivors as ARRAYS. Building a seed tuple here, converting
            # it to a delta tuple, then rebuilding an array in the cull's
            # kernel adapter was a pure round-trip: 47M generator-expression
            # calls plus 13.7s of adapter time per profiled run at depth 400.
            kept_ops.append(np.full(surv.shape[0], op, dtype=np.int32))
            kept_deltas.append(surv - tgt)

    global _V9_SEC_CULLED
    _V9_SEC_CULLED += n_bad
    # The exact residue filter (_tc_target_survives) is NOT applied here. At
    # depth it ran ~12,300 times per task and the cull then discarded 92% of
    # those candidates -- 2.16M calls and 104M dict lookups per profiled run,
    # the single largest cost after the cull itself. It now runs inside the
    # cull, on the ranked top-K only. Same final action list; ~12x fewer calls.
    if not kept_ops:
        return [], len(S), None, None
    ops_a = np.concatenate(kept_ops)
    del_a = np.concatenate(kept_deltas)
    # Dedup + deterministic order WITHOUT going through Python tuples: lexsort
    # takes its LAST key as most significant, so listing delta columns in
    # reverse and op last reproduces sorted() on (op, delta) exactly.
    keys = [del_a[:, t] for t in range(n_idx - 1, -1, -1)] + [ops_a]
    order = np.lexsort(keys)
    ops_a = ops_a[order]
    del_a = del_a[order]
    uniq = np.ones(ops_a.shape[0], dtype=bool)
    if ops_a.shape[0] > 1:
        uniq[1:] = ~((ops_a[1:] == ops_a[:-1])
                     & (del_a[1:] == del_a[:-1]).all(axis=1))
    ops_a = np.ascontiguousarray(ops_a[uniq])
    del_a = np.ascontiguousarray(del_a[uniq])
    # ONE tuple per surviving action, built once, already in final order.
    valid = [(int(ops_a[i]), tuple(map(int, del_a[i])))
             for i in range(ops_a.shape[0])]
    return valid, len(S), ops_a, del_a


def _tc_upenum_candidates_scalar(target, resolved_subs, env, reg, n_idx):
    """(op, delta) candidate superset from the store alone. Returns
    (set_of_actions, |S|).

    seed = T - shift puts T on a term POSITION, but the coefficient there is
    evaluated at the seed and usually comes out zero -- measured as 94-96% of
    this rule's overshoot. For a single-index coefficient that is exactly
    seed[k] == 0, i.e. T[k] == shift[k]: one integer compare, no equation
    built. Coefficients we cannot read this way are assumed non-zero, which
    only over-generates and so can never drop a real action."""
    tid = reg.get_id(target)
    S = [target]
    for k, sol in resolved_subs.items():
        if _tc_sol_contains(sol, target, tid):
            S.append(k)
    tbl = _tc_vanish_table(env)
    lin_all = _tc_lin_table(env, n_idx)
    # Positions where the TARGET is non-positive: a term is outside the
    # target's family exactly when one of these is positive for it. Computed
    # once per step instead of a bitmask per term.
    outside = [i for i in range(n_idx) if target[i] <= 0]
    thr = ibp_env.get_raw_strip_threshold()
    thr2 = tuple(thr)[:2] if thr is not None else None
    # Dedup on (op, seed), not (op, delta): they carry the same information
    # because the target is fixed for this step, and seed is what the sector
    # test needs. That way ONE tuple is built per (T, shift) pair instead of
    # two, and the delta -- needed only as the action's name -- is built just
    # for the survivors. zip beats index-based generators here.
    kept = []
    seen = set()
    n_bad = 0
    tbl_items = tbl.items()
    for T in S:
        for op, terms in tbl_items:
            lin_s = lin_all[op]
            for sh, ki, _u, _c in terms:
                if ki is not None and T[ki] == sh[ki]:
                    continue            # coefficient vanishes at this seed
                seed = tuple(map(operator.sub, T, sh))
                key = (op, seed)
                if key in seen:
                    continue
                seen.add(key)
                # Cheap test first: most rejects are banned integrals, and
                # that test builds nothing. Only survivors pay for the
                # coefficient sum.
                if not _tc_sector_ok(seed, terms, resolved_subs, outside,
                                     thr2, n_idx):
                    n_bad += 1
                elif _tc_target_survives(
                        seed, terms, resolved_subs, target, tid, n_idx,
                        ((lin_s[:, :n_idx] @ np.array(seed, dtype=np.int64)
                          + lin_s[:, n_idx]) % ibp_env.PRIME).tolist()):
                    kept.append(key)
                else:
                    n_bad += 1
    global _V9_SEC_CULLED
    _V9_SEC_CULLED += n_bad
    return ({(op, tuple(map(operator.sub, seed, target))) for op, seed in kept},
            len(S))


def _tc_sector_ok(seed, terms, resolved_subs, outside, thr2, n_idx):
    """Fast path. `outside` is the list of index positions where the TARGET is
    non-positive, precomputed once per step. sector_bitmask sets bit i iff
    index i is positive, so "introduces something outside the target's family"
    is exactly "some position in `outside` is positive here" -- a handful of
    integer compares, with NO tuple built and NO dict lookup.

    Most terms pass that test, so the expensive work (building the integral,
    checking whether it is a known integral that gets substituted away, and
    computing its weight for the strip test) only runs for the few that look
    bad. That ordering is the whole point: the previous version paid full
    price on every term.
    """
    for sh, ki, _u, _c in terms:
        if ki is not None and seed[ki] == 0:
            continue                    # coefficient vanishes: term absent
        for i in outside:
            if seed[i] + sh[i] > 0:
                break
        else:
            continue                    # stays inside the family
        k = tuple(map(operator.add, seed, sh))
        if k in resolved_subs:
            continue                    # replaced by its solution
        if thr2 is not None:
            w1 = w2 = 0
            for v in k:
                if v > 0:
                    w1 += v
                elif v < 0:
                    w2 -= v
            if (w1, w2) < thr2:
                continue                # stripped: never generated
        return False
    return True


_V9_VALID_ARRS = (None, None, None)     # (valid list, ops, deltas)


def _tc_upenum_valid(target, state, env):
    """Enumeration replacement. The generator already returns the actions in
    sorted, deduplicated order AND the matching numpy arrays, which are stashed
    so the cull's compiled kernel does not rebuild them from Python tuples."""
    global _V9_VALID_ARRS
    valid, _ns, ops_a, del_a = _tc_upenum_candidates(
        target, state.resolved_subs, env, _V7_REGISTRY, ibp_env.N_INDICES)
    _V9_VALID_ARRS = (valid, ops_a, del_a)
    return valid


# ── compiled cull kernel ───────────────────────────────────────────────────
try:
    from sailir._v9_cull_inner import cull_score as _CULL_C, PACK_MAX
except ImportError:                     # pure-Python fallback, as elsewhere
    _CULL_C = None
    PACK_MAX = 1 << 62
_V9_CULL_C_FALLBACKS = 0                # times the kernel could NOT be used
_V9_CULL_C = (_CULL_C is not None
              and os.environ.get('SAILIR_V9_CULL_C') not in ('0', 'off'))
_SHIFT_ARRS = None


def _v9_shift_arrays(env, n_idx):
    """Template offsets as one flat int32 array plus per-operator bounds."""
    global _SHIFT_ARRS
    if _SHIFT_ARRS is not None:
        return _SHIFT_ARRS
    n_ops = max(env.shifts) + 1
    off = np.zeros(n_ops + 1, dtype=np.int32)
    rows = []
    for op in range(n_ops):
        sh = env.shifts.get(op, ())
        off[op + 1] = off[op] + len(sh)
        rows.extend(sh)
    flat = (np.array(rows, dtype=np.int32) if rows
            else np.zeros((0, n_idx), dtype=np.int32))
    _SHIFT_ARRS = (flat, off)
    return _SHIFT_ARRS


def _v9_store_arrays(rsd, store_maxw, target, tid, n_idx, lo, mult,
                     mult_span):
    """Store as SORTED packed int64 keys + parallel (w1, w2, holds-target).

    Returns None if any key falls outside the declared per-index range, which
    would alias two distinct integrals onto one key.
    """
    m = len(rsd)
    keys = np.empty(m, dtype=np.int64)
    w1 = np.empty(m, dtype=np.int32)
    w2 = np.empty(m, dtype=np.int32)
    ht = np.zeros(m, dtype=np.uint8)
    i = 0
    for k, sol in rsd.items():
        acc = 0
        for t in range(n_idx):
            x = k[t] - lo[t]
            if x < 0 or x >= mult_span[t]:
                return None
            acc += x * mult[t]
        keys[i] = acc
        mw = store_maxw[k]
        w1[i] = mw[0]
        w2[i] = mw[1]
        ht[i] = 1 if target in sol else 0
        i += 1
    order = np.argsort(keys, kind='stable')
    return keys[order], w1[order], w2[order], ht[order]


def _v9_cull_scored_c(valid, target, state, env, store_maxw, rsd, n_idx):
    """Compiled scoring. Returns the SAME `scored` list the Python loop builds
    -- [((b, op, delta), ai), ...] in append order -- or None to fall back.

    Builds a MIXED-RADIX packing from the ranges this call can actually
    produce: every index of the target, of target+delta for every candidate,
    and of seed+offset for every template offset, plus every store key. A fixed
    range would reject legitimate targets (1_6_1_1_1_0_0_0_0_1 carries a 6, and
    a +2 offset reaches 8); a range that is too NARROW would alias two
    integrals onto one key and silently mis-score, so the bounds are computed
    conservatively and any overflow returns None.
    """
    global _V9_CULL_C_FALLBACKS
    n = len(valid)
    if not n:
        return []
    tgt = np.asarray(target, dtype=np.int32)
    cached_valid, cached_ops, cached_del = _V9_VALID_ARRS
    if cached_valid is valid and cached_ops is not None:
        ops = cached_ops                # built by the generator, no round-trip
        deltas = cached_del
    else:
        deltas = np.empty((n, n_idx), dtype=np.int32)
        ops = np.empty(n, dtype=np.int32)
        for i, (op, delta) in enumerate(valid):
            ops[i] = op
            deltas[i] = delta
    flat, off = _v9_shift_arrays(env, n_idx)

    # per-index bounds over everything the kernel can form
    seeds = tgt[None, :] + deltas
    lo_i = seeds.min(axis=0)
    hi_i = seeds.max(axis=0)
    if flat.shape[0]:
        lo_i = np.minimum(lo_i, (seeds.min(axis=0)[None, :] + flat).min(axis=0))
        hi_i = np.maximum(hi_i, (seeds.max(axis=0)[None, :] + flat).max(axis=0))
    if rsd:
        sk = np.array(list(rsd.keys()), dtype=np.int32)
        lo_i = np.minimum(lo_i, sk.min(axis=0))
        hi_i = np.maximum(hi_i, sk.max(axis=0))
    lo_i = np.minimum(lo_i, tgt)
    hi_i = np.maximum(hi_i, tgt)

    span = (hi_i - lo_i + 1).astype(np.int64)
    mult = np.empty(n_idx, dtype=np.int64)
    acc = 1
    for t in range(n_idx):
        mult[t] = acc
        acc *= int(span[t])
        if acc >= PACK_MAX:              # radix product will not fit int64
            _V9_CULL_C_FALLBACKS += 1
            return None

    st = _v9_store_arrays(rsd, store_maxw, target,
                          _V7_REGISTRY.get_id(target), n_idx,
                          lo_i, mult, span)
    if st is None:
        _V9_CULL_C_FALLBACKS += 1
        return None
    keys, w1, w2, ht = st
    keep, b1, b2 = _CULL_C(tgt, deltas, ops, flat, off, keys, w1, w2, ht,
                           lo_i.astype(np.int32), mult, n_idx)
    kl = keep.tolist()
    b1l = b1.tolist()
    b2l = b2.tolist()
    return [(((b1l[j], b2l[j]), valid[kl[j]][0], tuple(valid[kl[j]][1])),
             kl[j]) for j in range(len(kl))]


def _v9_cull(valid, target, state, env, max_actions):
    """Rank the enumerated actions by the store-only maxw bound, keep the top
    max_actions. Returns the kept list in RANK order -- the same order the
    training samples recorded, so a label index means the same thing here."""
    if not valid or not max_actions or max_actions <= 0:
        return valid
    n_idx = ibp_env.N_INDICES
    store_maxw, rsd = _v9_store_maxw(state.resolved_subs)
    W = _w12
    shifts = env.shifts
    scored = None
    if _V9_CULL_C:
        scored = _v9_cull_scored_c(valid, target, state, env, store_maxw, rsd,
                                   n_idx)
    if scored is None:
        scored = []
        for ai in range(len(valid)):
            op, delta = valid[ai]
            # map(operator.add, ...) beats an index-based generator expression by
            # ~2x on these 15-element vectors, and this runs ~4.5M times per step:
            # cProfile showed 71.6M generator-expression calls, 5.9s, in this loop
            # alone once weight() was fixed.
            seed = tuple(map(_ADD, target, delta))
            b = (0, 0)
            adm = False
            for sh in shifts[op]:
                k = tuple(map(_ADD, seed, sh))
                if k == target:
                    adm = True
                    continue                    # target is not in its own tail
                v = rsd.get(k)
                if v is not None:
                    if target in v:
                        adm = True
                    m = store_maxw[k]
                else:
                    m = W(k)
                if m > b:
                    b = m
            if adm:
                scored.append(((b, op, tuple(delta)), ai))
    if not scored:
        return []
    scored.sort(key=lambda e: e[0])
    if not _V9_UPENUM:
        return [valid[ai] for _k, ai in scored[:max_actions]]

    # STORE-ONLY path: the generator deferred the exact residue test (does the
    # target survive substitution with a non-zero coefficient?) because running
    # it on every candidate meant ~12,300 calls per task to keep 1,000. Apply
    # it here, walking the RANKED list and stopping once max_actions have
    # passed -- so the expensive test runs ~K times, not ~12K, and the surviving
    # list is identical to filtering first because the filter is independent of
    # the ranking key.
    tbl = _tc_vanish_table(env)
    lin = _tc_lin_table(env, n_idx)
    rs = state.resolved_subs
    tid = _V7_REGISTRY.get_id(target)
    P = ibp_env.PRIME
    out = []
    for _k, ai in scored:
        op, delta = valid[ai]
        seed = tuple(map(_ADD, target, delta))
        L = lin[op]
        sv = np.asarray(seed, dtype=np.int64)
        rc = ((L[:, :n_idx] @ sv + L[:, n_idx]) % P).tolist()
        if _tc_target_survives(seed, tbl[op], rs, target, tid, n_idx, rc):
            out.append(valid[ai])
            if len(out) >= max_actions:
                break
    return out


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


def _emit_dagger_rows(tasks, beam, probs, target_sector, step):
    """Write one jsonl training row per model-visited state.

    Schema matches data-gen/preprocess_to_tensors.py exactly so the existing
    packer consumes these unchanged. `subs` is emitted EMPTY: the p=101 corpus
    is packed without --keep-subs (the nosubs model ignores them), and the
    packer's include_subs=False path drops the field anyway.
    """
    global _DAGGER_FH
    import json as _dj
    if _DAGGER_FH is None:
        _DAGGER_FH = open(_DAGGER_OUT, 'a')
    N = ibp_env.N_INDICES
    for _ti, (_pi, _tgt, _valid) in enumerate(tasks):
        st = beam[_pi]
        _DAGGER_STATS['seen'] += 1
        used = {(o, tuple(t[i] + d[i] for i in range(N))) for (t, o, d) in st.path}
        lab = []
        for _j, (o, d) in enumerate(_valid):
            seed = tuple(_tgt[i] + d[i] for i in range(N))
            if (o, seed) in _CLOSURE_PROBE_SET and (o, seed) not in used:
                lab.append(_j)
        if not lab:
            _DAGGER_STATS['deadend'] += 1
            if _DAGGER_MISSING:
                _DAGGER_MISSING_SET.add(tuple(int(x) for x in _tgt))
            if not _DAGGER_DEADEND:
                continue
        # model's own pick, to decide whether this state is a mistake
        _top = int(probs[_ti, :len(_valid)].argmax()) if len(_valid) else -1
        if _DAGGER_MODE == 'errors' and lab and _top in lab:
            _DAGGER_STATS['skipped_model_right'] += 1
            continue
        row = {
            'scramble_id': -1,
            'sector_id': ibp_env.get_sector_id(_tgt),
            'sector_mask': [int(x) for x in target_sector],
            'step': step,
            'target': [int(x) for x in _tgt],
            'target_weight': [int(x) for x in ibp_env.weight(_tgt)[:2]],
            'start_target': [int(x) for x in _tgt],
            'expr': [[[int(x) for x in k], int(v)] for k, v in st.expr.items()],
            'subs': [],
            'valid_actions': [[int(o), [int(x) for x in d]] for (o, d) in _valid],
            'num_valid_actions': len(_valid),
            'chosen_action_idx': lab[0] if lab else 0,
            'valid_label_idxs': lab,
        }
        _DAGGER_FH.write(_dj.dumps(row) + '\n')
        _DAGGER_STATS['emitted'] += 1
    _DAGGER_FH.flush()


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
                # Restore the writer's registry BEFORE anything reads a
                # PackedEq. Older checkpoints have no such frame; those simply
                # cannot be resumed with packed substitutions.
                try:
                    _rf = pickle.load(f)
                except EOFError:
                    _rf = None
                if isinstance(_rf, dict) and '_registry' in _rf:
                    for _t in _rf['_registry']:
                        _V7_REGISTRY.get_id(tuple(_t))
                    print(f'  [resume] restored registry: '
                          f'{len(_rf["_registry"])} integrals', flush=True)
                elif _PACKED_RS:
                    raise RuntimeError(
                        'checkpoint has no _registry frame but packed '
                        'substitutions are ON -- its integral ids cannot be '
                        'interpreted. Re-produce the checkpoint.')
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
        # DUAL: seed BOTH lanes with the start state, so each sub-beam has its
        # own root and neither can be starved of ancestry by the other.
        beam = ([initial, initial._replace(lane=1)]
                if beam_sort == 'dual' else [initial])
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

    # SOFT w2 (2026-08-15). sort_weight is lexicographic on
    # (max_w12, n_non_masters, -score), so max_w12 is absolute and the model is
    # only a third-level tiebreak. That makes the weight lane structurally
    # unable to follow a truth path that goes temporarily UPHILL -- and the
    # truth-trace showed exactly that: at the step where the beam lost the truth
    # trajectory the correct child had max_w12 [11,2] against a cutoff of
    # [11,1], i.e. w1 tied and w2 worse by one, while being 0.97 nats BETTER on
    # the model and equal on n_non_masters. It was rejected on key #1 alone.
    #
    # SAILIR_MW2_PRICE=mu makes only w2 tradeable, at a price of mu nats per
    # unit:
    #     (w1,  mu*w2 + lambda*nm - score)
    # w1 keeps absolute priority, so the dot-count can still never worsen and
    # the search cannot ratchet along the dangerous axis. Every uphill move the
    # truth path was observed taking was in w2.
    #
    # A scalar collapse like w1+w2 would NOT do: (12,0) and (11,1) both sum to
    # 12 while lexicographically (11,1) < (12,0). The two axes are not
    # commensurate and must not be added.
    #
    # From the measured step: the truth child survives iff mu*2 + 3.35 <
    # mu*1 + 4.32, i.e. mu < 0.97.
    # Default 0 = disabled = bit-identical to the previous behaviour.
    def _norm_score(s):
        # match sort_prob's normalisation so mu means the same thing at any
        # SAILIR_SCORE setting (identical to s.score under the default 'local')
        if _SCORE_MODE == 'cumulative':
            return s.score / max(len(s.path), 1)
        if _SCORE_MODE == 'decay':
            return ((1.0 - _SCORE_DECAY) * s.score if _SCORE_DECAY < 1.0
                    else s.score / max(len(s.path), 1))
        return s.score

    def sort_weight(s):
        if _MW2_PRICE > 0.0:
            # w2 - mu*score, with NO nm term. Order-equivalent to the earlier
            # mu*w2 - score under mu -> 1/mu, so this is a reparametrisation
            # rather than a new rule -- kept because it makes mu read directly
            # as "nats of model log-prob per unit of w2".
            # SINGLE-KEY BEAM: (w1, mu*w2 + lambda*nm - score).
            # Measured at the step where the truth trajectory is lost
            # (2,1,0 depth 3), truth child vs the binding cutoff:
            #     truth  w1=11 w2=2 nm=5   score=-2.77
            #     cutoff w1=11 w2=1 nm=15  score=-1.77
            # The truth child is WORSE on w2 (+1) and on score (+1.0 nat) but
            # far better on nm (5 vs 15). With no nm term it therefore loses at
            # every mu >= 0 -- verified at mu = 0.01, 0.05, 1, 5. Restoring nm:
            #     truth wins iff  lambda*(15-5) > mu + 1.0,  i.e. lambda > (mu+1)/10
            # so lambda ~ 0.15-0.3 at small mu, just above the 0.1 the beam has
            # been running.
            return (s.max_w12[0],
                    _MW2_PRICE * s.max_w12[1]
                    + _nm_penalty * s.n_non_masters - _norm_score(s))
        return (s.max_w12, s.n_non_masters, -s.score)

    def sort_totalweight(s):
        return (s.total_w12, s.n_non_masters, -s.score)

    # Drain pressure for prob-sort (2026-08-03): pure local log-prob has no
    # incentive to shrink the active bucket, so on long runs the beam wanders
    # and nm ratchets into the hundreds (frozen-7 divergence). Penalize each
    # active non-master by SAILIR_NM_PENALTY in log-prob units so states that
    # grow the bucket must buy it with genuinely better-ranked actions.
    _nm_penalty = float(os.environ.get('SAILIR_NM_PENALTY', '0.0'))

    def sort_prob(s):
        # model-credited selection (2026-08-02): action log-prob first
        # (minus the nm drain penalty) — lets a confident learned route
        # survive weight-uphill stretches that sort_weight prunes
        # structurally. Weight as tiebreak.
        #
        # Under SAILIR_SCORE=cumulative the score is a running SUM, which grows
        # to ~-4000 by step 4000 while the nm penalty stays ~-2 -- the penalty
        # would silently become 0.05% of the ranking and stop working. Dividing
        # by the path length restores its relative weight. It does NOT change
        # the score ordering: the beam advances in lockstep, so every state
        # being compared has the same path length and is divided by the same k.
        if _SCORE_MODE == 'cumulative':
            _sc = s.score / max(len(s.path), 1)
        elif _SCORE_MODE == 'decay':
            # (1-gamma) turns the geometric sum into a WEIGHTED MEAN of recent
            # log-probs, so the scale matches 'local' (and gamma=0 is exactly
            # local) and the nm penalty keeps its relative weight at any gamma.
            _sc = (1.0 - _SCORE_DECAY) * s.score if _SCORE_DECAY < 1.0 \
                else s.score / max(len(s.path), 1)
        else:
            _sc = s.score
        if _W1_FIRST:
            return (s.max_w12[0],
                    -(_sc - _nm_penalty * s.n_non_masters),
                    s.max_w12[1], s.n_non_masters)
        return (-(_sc - _nm_penalty * s.n_non_masters),
                s.max_w12, s.n_non_masters)

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
    # v8 diagnostics: PRE-CAP action-space accounting. _PRECAP_STATS per step:
    # [n_tasks, total_precap, max_precap, n_capped_tasks]. SAILIR_PRECAP_DUMP
    # additionally dumps each task's FULL pre-cap list per step.
    _precap_dump_dir = os.environ.get('SAILIR_PRECAP_DUMP')
    if _precap_dump_dir:
        os.makedirs(_precap_dump_dir, exist_ok=True)
    _PRECAP_STATS = [0, 0, 0, 0]
    _PRECAP_BUF = []

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
            tied = [min(nm, key=_target_key)]  # v8: SINGLE full-order target
            dummy_subs = {k: {} for k in s.resolved_subs.keys()}
            if _V9_UPENUM:
                indirect_cache = None   # cu is never built
            elif s.aux_flat is not None:
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
                if _V9_UPENUM:
                    valid = _tc_upenum_valid(target, s, env)
                else:
                    valid = _packed_cu.enumerate_valid_actions_with_indirect_cache_packed(
                        target, indirect_cache, s.resolved_subs,
                        env.ibp_t, env.li_t, env.shifts, 'subsector',
                        env._raw_eq_cache, _V7_REGISTRY, ibp_env.N_INDICES,
                    )
                if _trace_fh is not None:
                    _truth_enum_check('A', s, target, valid)
                if (not _bounded_tabu_on
                        and tabu_dict is not None and valid):
                    expr_fp = frozenset(s.expr.items())
                    tabu_set = tabu_dict.get(expr_fp)
                    if tabu_set:
                        _pre_tabu = list(valid)
                        valid = [(op, dlt) for (op, dlt) in valid
                                 if (target, op, dlt) not in tabu_set]
                        # Did tabu just ban the truth action? The tabu key is
                        # the EXPRESSION, but a state is (expr, subs) -- so an
                        # action tried by a different state that merely shares
                        # this expression can ban it here.
                        if _trace_fh is not None and _truth_actions:
                            _tn = len(s.path)
                            if (_tn < len(_truth_actions) and
                                all(s.path[_i] == _truth_actions[_i]
                                    for _i in range(_tn))):
                                _tta = _truth_actions[_tn]
                                if (tuple(target) == _tta[0]
                                        and (_tta[1], _tta[2]) in _pre_tabu
                                        and (_tta[1], _tta[2]) not in valid):
                                    import json as _j
                                    _trace_fh.write(_j.dumps(dict(
                                        event='TABU_BANNED_TRUTH', depth=_tn,
                                        n_before=len(_pre_tabu),
                                        n_after=len(valid),
                                        tabu_size=len(tabu_set))) + chr(10))
                                    _trace_fh.flush()
                elif _bounded_tabu_on and valid:
                    _bt_key = (frozenset(s.expr.items()), target)
                    _bt_set = _bt_tabu.get(_bt_key)
                    if _bt_set:
                        _nblk = sum(1 for a in valid if a in _bt_set)
                        if _nblk < len(valid) and _nblk <= _tabu_cap_eff:
                            valid = [a for a in valid if a not in _bt_set]
                if valid:
                    if _V9_CULL:
                        valid = _v9_cull(valid, target, s, env, max_actions)
                    elif _ACTION_SELECT != 'first900':
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

    if _TRUTH_TRACE and _trace_fh is None:
        _truth_trace_load()
    for step in range(resume_step, max_steps):
        _trace_parents = list(beam) if _trace_fh is not None else ()
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

        # TRUTH-PATH BEAM CENSUS (SAILIR_TRUTH_CENSUS=1). Written BEFORE the
        # macro dedup and before enumeration, so a truth state that is in the
        # beam but never reaches the enumerator is visible. Without this, the
        # trace jumps from "on the truth path" to "child never generated" with
        # nothing in between -- which is exactly where a depth-4 loss hid.
        if (_trace_fh is not None and _truth_actions
                and os.environ.get('SAILIR_TRUTH_CENSUS') == '1'):
            import json as _jc

            def _dc(st):
                n = len(st.path)
                return n if (n <= len(_truth_actions) and
                             all(st.path[i] == _truth_actions[i]
                                 for i in range(n))) else -1
            _dd = [_dc(x) for x in beam]
            _on = [i for i, d in enumerate(_dd) if d >= 0]
            _trace_fh.write(_jc.dumps(dict(
                event='BEAM_CENSUS', step=step, n_beam=len(beam),
                depths_on_truth=[_dd[i] for i in _on],
                beam_idx_on_truth=_on,
                max_depth=max(_dd) if _dd else -1)) + chr(10))
            _trace_fh.flush()

        # ── v6: macro-beam dedup ──────────────────────────────────────────
        # Collapse beam slots that share the same expr_fp. The model only
        # sees expr + RS_KEYS (not values; see prepare_batched_input_v5_dummy)
        # so duplicate-expr states return the same model scores → waste.
        # Keep best variant per expr_fp by progress key (low max_w12, low nm,
        # high score).
        _dedup_t = time.time()
        # DUAL: key by (expr_fp, lane), not expr_fp alone. Both lanes start
        # from the SAME expression, so a plain expr_fp key would keep one
        # representative, expand only that lane, and the other lane -- whose
        # children inherit the parent's lane -- would receive no candidates at
        # all and vanish on the first step.
        # Same key as the early dedup (SAILIR_DEDUP_KEY, default 'exprpath'):
        # a state is (expr, subs), and the path distinguishes stores without
        # materialising them. Fixing only the early copy moved the kill here --
        # measured on 1,0,1,0,0,0,5,1,1,1, where the truth state survived to
        # depth 4 and was then deleted by THIS dedup at the next step.
        _dedup_key_top = os.environ.get('SAILIR_DEDUP_KEY', 'exprpath')
        macro_groups = {}  # key → best State
        for s in beam:
            efp = frozenset(s.expr.items())
            if _dedup_key_top == 'exprpath':
                efp = (efp, frozenset(s.path))
            if beam_sort == 'dual':
                efp = (efp, s.lane)
            cur = macro_groups.get(efp)
            if cur is None or (s.max_w12, s.n_non_masters, -s.score) < \
                              (cur.max_w12, cur.n_non_masters, -cur.score):
                macro_groups[efp] = s
        n_orig_beam = len(beam)
        _pre_dedup = beam
        beam = list(macro_groups.values())
        # DEDUP CHECK: macro-dedup keys on the EXPRESSION alone, but a state is
        # (expr, subs). Two states with the same expression and different
        # substitution stores are not duplicates, yet one is discarded. Record
        # whether the truth-path state went in and did not come out.
        if _trace_fh is not None and _truth_actions:
            def _d(st):
                n = len(st.path)
                return n if (n <= len(_truth_actions) and
                             all(st.path[i] == _truth_actions[i]
                                 for i in range(n))) else -1
            _pre = max((_d(x) for x in _pre_dedup), default=-1)
            _post = max((_d(x) for x in beam), default=-1)
            if _pre >= 0 and _post < 0:
                import json as _j
                _trace_fh.write(_j.dumps(dict(
                    step=step, event='DEDUP_KILLED_TRUTH', depth=_pre,
                    n_before=n_orig_beam, n_after=len(beam))) + chr(10))
                _trace_fh.flush()
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
                tied = [min(nm, key=_target_key)]  # v8: SINGLE full-order target
                if _v5_prof:
                    _p1['nm_tied'] += time.time() - _t
                # Dummy subs (RS keys with empty values) — only used as keys for
                # enumerate_valid_actions iteration (it reads keys, not values).
                dummy_subs = {k: {} for k in s.resolved_subs.keys()}
                # Use cached aux_flat if available, else rebuild from scratch.
                _t = time.time() if _v5_prof else 0
                if _V9_UPENUM:
                    indirect_cache = None   # cu is never built
                elif s.aux_flat is not None:
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
                    _ct['n_iraws_total'] += len(indirect_cache or ())
                    if s.aux_flat:
                        _ct['cu_size_max'] = max(_ct['cu_size_max'],
                                                  len(s.aux_flat[0]))
                    _ct['rs_max'] = max(_ct['rs_max'], len(s.resolved_subs))
                for target in tied:
                    _t = time.time() if _v5_prof else 0
                    if _V9_UPENUM:
                        valid = _tc_upenum_valid(target, s, env)
                    else:
                        valid = _packed_cu.enumerate_valid_actions_with_indirect_cache_packed(
                            target, indirect_cache, s.resolved_subs,
                            env.ibp_t, env.li_t, env.shifts, 'subsector',
                            env._raw_eq_cache, _V7_REGISTRY, ibp_env.N_INDICES,
                        )
                    if _trace_fh is not None:
                        _truth_enum_check('B', s, target, valid)
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
                                _pre_bt = list(valid)
                                if _SOL_TABU:
                                    # Re-derive the substitution for the BLOCKED
                                    # minority only, using exactly the path
                                    # apply_action_v5 uses -- aux cache when the
                                    # action is indirect, raw equation + this
                                    # state's store when it is DIRECT. The
                                    # aux-only version kept the ban on every
                                    # direct action, which is precisely the case
                                    # this exists to rescue.
                                    _N = ibp_env.N_INDICES
                                    _efp = frozenset(s.expr.items())
                                    _keep = []
                                    for _a in valid:
                                        if _a not in _bt_set:
                                            _keep.append(_a); continue
                                        _op, _dl = _a
                                        _sd = tuple(target[_i] + _dl[_i]
                                                    for _i in range(_N))
                                        _cch = None
                                        if s.aux_flat is not None:
                                            _ix = s.aux_flat[2].get((_op, _sd))
                                            if _ix is not None:
                                                _cch = _v7_as_dict(
                                                    s.aux_flat[0][_ix])
                                        if _cch is None:
                                            _rw = env.get_raw_equation_cached(
                                                _op, _sd)
                                            _cch = (
                                                apply_resolved_subs_dict_x_packed(
                                                    _rw, s.resolved_subs,
                                                    _V7_REGISTRY, ibp_env.PRIME)
                                                if _PACKED_RS else
                                                apply_resolved_subs(
                                                    _rw, s.resolved_subs))
                                        _isT = (_truth_actions and
                                                len(s.path) < len(_truth_actions)
                                                and all(s.path[_q] ==
                                                        _truth_actions[_q]
                                                        for _q in range(len(s.path)))
                                                and (tuple(target),) ==
                                                (_truth_actions[len(s.path)][0],)
                                                and _op ==
                                                _truth_actions[len(s.path)][1]
                                                and tuple(_dl) ==
                                                _truth_actions[len(s.path)][2])
                                        if (target not in _cch
                                                or _cch[target] == 0):
                                            if _isT: _SOL_WHY[0] = 'no_target'
                                            continue
                                        _sp = solve_ibp_for(_cch, target)
                                        if _sp is None:
                                            if _isT: _SOL_WHY[0] = 'unsolvable'
                                            continue
                                        _fp = hash(frozenset(_sp.items()))
                                        _seen = _SOL_FP.get(
                                            (_efp, tuple(target), _op,
                                             tuple(_dl)))
                                        if _isT:
                                            _SOL_WHY[0] = (
                                                'fp_seen' if (_seen is not None
                                                              and _fp in _seen)
                                                else 'UNBANNED')
                                            _SOL_WHY[1] = (len(_seen)
                                                           if _seen else 0)
                                            _wr = _SOL_WHO.get(
                                                (_efp, tuple(target), _op,
                                                 tuple(_dl)), [])
                                            _mine = tuple(s.path)
                                            _anc = any(
                                                len(w) <= len(_mine) and
                                                _mine[:len(w)] == w
                                                for w in _wr)
                                            _SOL_WHY.append(
                                                f'writers={len(_wr)} '
                                                f'ancestor_of_me={_anc} '
                                                f'writer_depths='
                                                f'{sorted(len(w) for w in _wr)}')
                                        if _seen is None or _fp not in _seen:
                                            _keep.append(_a)   # different sub
                                    valid = _keep
                                elif _LINEAGE_TABU:
                                    _mypath = tuple(s.path)
                                    _mylen = len(_mypath)
                                    _efp2 = frozenset(s.expr.items())
                                    _pfx = {}      # len -> hash(prefix), memoised
                                    def _anc(_act):
                                        _w = _BT_WHO.get(
                                            (_efp2, target, _act))
                                        if not _w:
                                            return False
                                        for _L, _H in _w:
                                            if _L > _mylen:
                                                continue
                                            _h = _pfx.get(_L)
                                            if _h is None:
                                                _h = hash(_mypath[:_L])
                                                _pfx[_L] = _h
                                            if _h == _H:
                                                return True
                                        return False
                                    valid = [a for a in valid
                                             if a not in _bt_set or not _anc(a)]
                                else:
                                    valid = [a for a in valid
                                             if a not in _bt_set]
                                # BOUNDED TABU keys on (expression, target) --
                                # NOT on the full state. Two states with the
                                # same expression and different substitution
                                # stores are different states with different
                                # consequences, yet an action tried by one is
                                # banned for the other.
                                if _trace_fh is not None and _truth_actions:
                                    _bn = len(s.path)
                                    if (_bn < len(_truth_actions) and
                                        all(s.path[_bi] == _truth_actions[_bi]
                                            for _bi in range(_bn))):
                                        _bta = _truth_actions[_bn]
                                        if (tuple(target) == _bta[0]
                                            and (_bta[1], _bta[2]) in _pre_bt
                                            and (_bta[1], _bta[2]) not in valid):
                                            import json as _jb
                                            _trace_fh.write(_jb.dumps(dict(
                                                event='BOUNDED_TABU_BANNED_TRUTH',
                                                depth=_bn, why=_SOL_WHY[0],
                                                n_seen_fps=_SOL_WHY[1],
                                                writer=(_SOL_WHY[2]
                                                        if len(_SOL_WHY) > 2
                                                        else '?'),
                                                n_before=len(_pre_bt),
                                                n_after=len(valid),
                                                n_blocked=int(_nblk))) + chr(10))
                                            _trace_fh.flush()
                        if _v5_prof:
                            _p1['tabu'] += time.time() - _t
                    if valid:
                        _PRECAP_STATS[0] += 1
                        _PRECAP_STATS[1] += len(valid)
                        _PRECAP_STATS[2] = max(_PRECAP_STATS[2], len(valid))
                        if len(valid) > max_actions:
                            _PRECAP_STATS[3] += 1
                        if _precap_dump_dir:
                            _PRECAP_BUF.append((tuple(target), list(valid)))
                        if _V9_CULL:
                            valid = _v9_cull(valid, target, s, env, max_actions)
                            if _V9_ORACLE is not None and _v9_on_truth_path(s):
                                # ONLY the state still on the recorded
                                # trajectory is comparable: it alone has the
                                # store the corpus was generated with. Verify
                                # its culled space, then force the oracle action
                                # so it stays on-path and later steps remain
                                # checkable. Divergent beam states are left to
                                # the model and are not checked.
                                _st = _V9_ORACLE_STATS
                                _st['deepest_on_path'] = max(
                                    _st.get('deepest_on_path', 0), len(s.path))
                                _oa = _v9_oracle_check(len(s.path), target, valid)
                                if _oa is not None:
                                    if _V9_ORACLE_RANK:
                                        # Keep the FULL space so the model
                                        # scores it; record where the truth
                                        # action sits in the cull's ranking,
                                        # and mark this task so the model rank
                                        # can be read off after the forward.
                                        try:
                                            _cr = valid.index(_oa)
                                        except ValueError:
                                            _cr = -1
                                        _V9_RANK_PENDING[len(tasks)] = (
                                            len(s.path), _oa, _cr, len(valid))
                                    else:
                                        valid = [_oa]
                        elif _ACTION_SELECT != 'first900':
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
                    tied = [min(nm, key=_target_key)]  # v8: SINGLE full-order target
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
                        tied = [min(nm, key=_target_key)]  # v8: SINGLE full-order target
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
        # ORACLE mode has no model: the action was already forced to the single
        # recorded truth action, so a uniform score is enough to carry it
        # through selection. Skipping the forward is what lets the alignment
        # test run without a checkpoint.
        _skip_model = model is None
        for chunk_start in (() if _skip_model
                            else range(0, len(batch_data), chunk_sz)):
            chunk = batch_data[chunk_start:chunk_start + chunk_sz]
            # Cap chunk's max_actions to the global so the slice assign below
            # has consistent width.
            b = prepare_batched_input_v5_dummy(chunk, device,
                                                max_actions=global_max_actions_eff)
            # EQACT: the model also consumes each action's RESOLVED equation.
            # There is no precompute at search time -- the beam visits states no
            # corpus contains -- so these are resolved from the CURRENT store,
            # by the SAME EqResolver the training dataloader used. The expression
            # is truncated to SAILIR_EXPR_TERMS here for the same reason the
            # equations are capped at SAILIR_EQ_TERMS inside the resolver:
            # training saw both cut, and an uncut input is out of distribution.
            _eqkw = {}
            if _eqi_mod().is_eqact(model):
                _eqkw = _eqi_mod().build_eq_tensors(
                    # PAD TO THE WIDTH `b` ACTUALLY BUILT, not the global max:
                    # prepare_batched_input_v5_dummy sizes to
                    # min(this chunk's max actions, global), so a chunk whose
                    # widest state has fewer actions yields a NARROWER tensor.
                    # Using the global here mismatched the action axis (464 vs
                    # 401) and the model refused to broadcast.
                    chunk, _V7_REGISTRY, env, env.ibp_t, env.li_t,
                    ibp_env.N_INDICES, b['action_ibp_ops'].shape[1], device)
                b = _eqi_mod().truncate_expr(b, None)
                if getattr(model, 'use_start_target', False):
                    # The START target T, NOT the current one. They differ:
                    # measured on the corpus, target == T on only 0.9% of rows
                    # (step 0), median |target - T| = 6 elsewhere. _START_INT is
                    # set once from the run's launch integral, which is exactly
                    # what `start_target` means in the training rows
                    # (preprocess_to_tensors.py:198).
                    # _START_INT is only set by _init_sym_drop, which this
                    # path need not call. The launch integral is recoverable
                    # without new plumbing: the worker starts the run with
                    # start_expr = {start_int: 1} (onestep_worker_v9.py:370),
                    # so a single-key start_expr IS the start target. Anything
                    # else is not a plain single-integral launch and we refuse
                    # rather than guess.
                    _sint = _START_INT
                    if _sint is None and len(start_expr) == 1:
                        _sint = next(iter(start_expr))
                    if _sint is None:
                        raise RuntimeError(
                            'eqact model has use_start_target=True but the '
                            'start target is unknown (_START_INT unset and '
                            f'start_expr has {len(start_expr)} keys) -- '
                            'refusing to substitute the current target, which '
                            'would feed the model a different input than it '
                            'trained on.')
                    _st = torch.tensor(tuple(_sint), dtype=torch.long,
                                       device=device)
                    _eqkw['start_target_integral'] = _st.unsqueeze(0).expand(
                        b['target_integral'].shape[0], -1)
            with torch.no_grad():
                _, chunk_probs = model(
                    b['expr_integrals'], b['expr_coeffs'], b['expr_mask'],
                    b['sub_keys'], b['sub_repl_ints'], b['sub_repl_coeffs'],
                    b['sub_repl_mask'], b['sub_mask'],
                    b['action_ibp_ops'], b['action_deltas'], b['action_mask'],
                    b['sector_mask'], b['target_integral'], **_eqkw,
                )
            cn = chunk_probs.shape[1]
            probs_all[chunk_start:chunk_start + len(chunk), :cn] = chunk_probs
            del b, chunk_probs
        probs = probs_all
        if _CLOSURE_PROBE_SET is not None:
            _N = ibp_env.N_INDICES
            _avail = []
            for _pi, _tgt, _valid in tasks:
                _st = beam[_pi]
                _used = {(_o, tuple(_t[i] + _d[i] for i in range(_N)))
                         for (_t, _o, _d) in _st.path}
                _c = 0
                for (_o, _d) in _valid:
                    _seed = tuple(_tgt[i] + _d[i] for i in range(_N))
                    if (_o, _seed) in _CLOSURE_PROBE_SET and (_o, _seed) not in _used:
                        _c += 1
                _avail.append(_c)
            if _DAGGER_OUT is not None:
                _emit_dagger_rows(tasks, beam, probs, target_sector, step)
            if _avail:
                _z = sum(1 for a in _avail if a == 0)
                print(f'  [CLOSUREPROBE] step={step} tasks={len(_avail)} '
                      f'expert_can_label={len(_avail)-_z} exhausted={_z} '
                      f'frac_labelable={(len(_avail)-_z)/len(_avail):.3f} '
                      f'mean_avail_rows={sum(_avail)/len(_avail):.1f} '
                      f'max_avail={max(_avail)}', flush=True)
        # OFF-MANIFOLD CONFIDENCE PROBE (SAILIR_CONF_PROBE=1). Top model score
        # per (parent,target) task, logged BEFORE beam selection so the sort
        # cannot bias the sample. Training states all lie on a truth reduction
        # path; beam states past step ~1 do not. v9 applies the same fastmaxw
        # cull the corpus was generated with, so these scores are directly
        # comparable to on-manifold scores measured on val states.
        if os.environ.get('SAILIR_CONF_PROBE') == '1':
            with torch.no_grad():
                _pm = probs.max(dim=1).values.tolist()
            if _pm:
                _sm = sorted(_pm); _n = len(_sm)
                print(f'  [CONFPROBE] step={step} n_tasks={_n} '
                      f'max={_sm[-1]:.6f} p90={_sm[int(0.9*(_n-1))]:.6f} '
                      f'median={_sm[_n//2]:.6f} p10={_sm[int(0.1*(_n-1))]:.6f} '
                      f'min={_sm[0]:.6f} '
                      f'frac>0.99={sum(v>0.99 for v in _sm)/_n:.3f}', flush=True)
        if _v5_prof:
            _p2['model_fwd'] += time.time() - _t

        # For each task, take top-K actions by prob, apply, generate candidates
        # SAILIR_TOP_K decouples K from beam_width. They were entangled --
        # K = beam_width//2 -- so the DUAL mode (beam 80 = 2 lanes x 40) would
        # otherwise silently expand 40 actions per state instead of 20.
        _envk = os.environ.get('SAILIR_TOP_K')
        if _envk:
            K = max(1, int(_envk))
        else:
            K = max(1, top_k if top_k is not None else beam_width // 2)
        _pick_dump_step = [] if _pick_dump_dir else None
        for ti, (parent_idx, target, valid) in enumerate(tasks):
            _force_only = None      # set when the oracle forces this task
            n_v = min(len(valid), probs.shape[1])
            row = probs[ti, :n_v].numpy()
            if _V9_ORACLE_RANK and ti in _V9_RANK_PENDING:
                _dep, _oa, _cull_rank, _nsp = _V9_RANK_PENDING.pop(ti)
                try:
                    _ai = valid.index(_oa)
                except ValueError:
                    _ai = -1
                if _ai >= 0 and _ai < n_v:
                    # rank = how many actions the model scores ABOVE the truth
                    _mrank = int((row > row[_ai]).sum())
                    _mprob = float(row[_ai])
                    _top = float(row.max())
                else:
                    _mrank, _mprob, _top = -1, 0.0, float(row.max())
                _V9_RANKS.append((_dep, _cull_rank, _mrank, _nsp, _mprob,
                                  _top))
                print(f'[rank] depth={_dep} space={_nsp} '
                      f'cull_rank={_cull_rank} model_rank={_mrank} '
                      f'p_truth={_mprob:.4g} p_top={_top:.4g}', flush=True)
                # Now FORCE the truth action for the transition. Measuring the
                # rank requires the model to score the full space, but letting
                # it then pick freely takes the beam off the truth path (it
                # left at step 2), and there is nothing to measure once off it.
                # One-hot the row so top-K can only choose the truth action.
                if _ai >= 0 and _ai < n_v:
                    row = np.zeros_like(row)
                    row[_ai] = 1.0
                    _force_only = int(_ai)
            if _ACTION_SCORE != 'model':
                # Overridden HERE, before top-K/tabu/scoring, so the only thing
                # that changes is where the scores come from.
                row = _model_free_row(target, valid, n_v)
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
                    if _LINEAGE_TABU:
                        # Same ancestry rule as the pre-filter. Tabu is applied
                        # in TWO places -- here and at the candidate filter --
                        # so scoping only one leaves the action blocked here and
                        # the fix silently does nothing.
                        _mp = tuple(parent_state.path)
                        _ml = len(_mp)
                        _pf = {}
                        def _anc2(_act):
                            _w = _BT_WHO.get((_key_bt[0], _key_bt[1], _act))
                            if not _w:
                                return False
                            for _L, _H in _w:
                                if _L > _ml:
                                    continue
                                _h = _pf.get(_L)
                                if _h is None:
                                    _h = hash(_mp[:_L])
                                    _pf[_L] = _h
                                if _h == _H:
                                    return True
                            return False
                        _blk = [i for i in range(n_v)
                                if valid[i] in _tset and _anc2(valid[i])]
                    else:
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
                _stamp = (len(parent_state.path),
                          hash(tuple(parent_state.path))) if _LINEAGE_TABU else None
                for _ai in top_idx:
                    _a2 = valid[int(_ai)]
                    _ts2.add(_a2)
                    if _LINEAGE_TABU:
                        _BT_WHO.setdefault(
                            (_key_bt[0], _key_bt[1], _a2), set()).add(_stamp)
            else:
                # Take top-K by prob
                top_idx = np.argsort(-row)[:K]
                _n_blocked = -1
            # ORACLE FORCING. One-hotting `row` above only makes the truth
            # action rank FIRST; top-K still expands K actions, so K-1
            # zero-probability siblings are expanded too. The beam then selects
            # by WEIGHT, not by model score, and can keep one of those siblings
            # and drop the truth child -- silently leaving the recorded path,
            # after which there is nothing meaningful left to measure.
            #
            # MEASURED: in rank mode every step expanded cand=20 and the walk
            # left the truth path at step 1 on 1_2_1_1_1_1_1_1_0_1 (1 step
            # measured of 338). True forcing (valid=[oa], which is what rank
            # mode does NOT do) expands cand=1 and tracks 338/338. Two other
            # targets only looked fine because the model's pick happened to be
            # weight-best at every step, so the walk reproduced truth by luck.
            #
            # Restrict the expansion to the truth action itself. Only affects
            # runs with SAILIR_V9_ORACLE_RANK; _force_only stays None otherwise.
            if _force_only is not None:
                top_idx = np.array([_force_only], dtype=int)
            # TRUTH SCORE: what does the model actually give the truth action?
            # `row` holds the model probabilities over `valid`, so this is the
            # first place the real ranking exists -- the index inside `valid` is
            # only enumeration order and says nothing about the model.
            if _trace_fh is not None and _truth_actions:
                _n = len(parent_state.path)
                if (_n < len(_truth_actions) and
                        all(parent_state.path[_i] == _truth_actions[_i]
                            for _i in range(_n))):
                    _ta = _truth_actions[_n]
                    if tuple(target) != _ta[0]:
                        import json as _j2
                        _trace_fh.write(_j2.dumps(dict(
                            event='TRUTH_TARGET_MISMATCH_AT_SCORER', depth=_n,
                            beam_target=list(target),
                            truth_target=list(_ta[0]))) + chr(10))
                        _trace_fh.flush()
                    if tuple(target) == _ta[0]:
                        try:
                            _vi = list(valid).index((_ta[1], _ta[2]))
                        except ValueError:
                            _vi = -1
                        import json as _j
                        if _vi < 0:
                            # SILENT BEFORE: the one case that matters most --
                            # the truth action is not even in the enumerated
                            # set, so it can never be ranked, expanded or
                            # selected. Writing nothing here made a depth-4
                            # enumeration failure indistinguishable from the
                            # scorer never running.
                            _trace_fh.write(_j.dumps(dict(
                                event='TRUTH_NOT_ENUMERATED', depth=_n,
                                target=list(target),
                                truth_op=int(_ta[1]),
                                truth_delta=list(_ta[2]),
                                n_valid=len(valid))) + chr(10))
                            _trace_fh.flush()
                        if _vi >= 0:
                            _ordr = np.argsort(-row)
                            _rank = int(np.where(_ordr == _vi)[0][0])
                            # WHAT BEAT IT. When the truth action is demoted,
                            # the useful question is whether the model chose
                            # another action that is ALSO on the truth walk --
                            # i.e. a legitimate alternative rather than a
                            # mistake. Records the top-1 action and whether that
                            # (op, delta) appears anywhere in the truth action
                            # sequence, and if so at what depth.
                            _top_i = int(_ordr[0])
                            _top_act = (list(valid)[_top_i]
                                        if _top_i < len(valid) else None)
                            _top_in_truth = -1
                            if _top_act is not None:
                                for _k, _tk in enumerate(_truth_actions):
                                    if (_tk[1], _tk[2]) == (_top_act[0],
                                                            _top_act[1]):
                                        _top_in_truth = _k
                                        break
                            _trace_fh.write(_j.dumps(dict(
                                event='TRUTH_TOP1', depth=_n,
                                truth_rank=_rank,
                                top_op=(int(_top_act[0])
                                        if _top_act else None),
                                top_delta=(list(_top_act[1])
                                           if _top_act else None),
                                top_action_is_truth_action_at_depth=_top_in_truth,
                                n_valid=len(valid))) + chr(10))
                            _trace_fh.flush()
                            _trace_fh.write(_j.dumps(dict(
                                event='TRUTH_SCORE', depth=_n,
                                model_rank=_rank, K=int(K), n_valid=int(n_v),
                                prob=float(row[_vi]),
                                top_prob=float(row[_ordr[0]]),
                                expanded=bool(_rank < K))) + chr(10))
                        else:
                            # reached the scorer but the action is GONE from
                            # `valid` by this point -- distinguishes "dropped
                            # somewhere between enumeration and scoring" from
                            # "this code path was never reached at all"
                            _trace_fh.write(_j.dumps(dict(
                                event='TRUTH_MISSING_AT_SCORER', depth=_n,
                                n_valid=int(n_v))) + chr(10))
                        _trace_fh.flush()
            # WAS A CLOSURE ACTION EVEN AVAILABLE HERE? For a parent that is
            # still inside the library, find every enumerated action that is a
            # library row for THIS target, and where the model ranked the best
            # of them. Distinguishes the two readings of a library death:
            #   avail=0            -> the state was structurally dead already;
            #                         the real mistake was culling diversity
            #                         earlier, and the fix is the beam.
            #   avail>0, rank>=K   -> closure actions existed and the model
            #                         ranked none inside the expanded top-K;
            #                         the fix is the model.
            # NOTE this also tests an assumption I had been making: library
            # membership does NOT guarantee a completable walk. The recorder
            # only ever picks rows that act on the CURRENT target, but the beam
            # chooses its own target, so a library-consistent state can reach a
            # target none of its remaining rows act on.
            if _LIB is not None:
                _q = np.asarray(row[:len(valid)], dtype=np.float64)
                _q = _q[_q > 0]
                _ALL_PARENT.append(dict(
                    in_lib=bool(_in_library(parent_state)),
                    depth=len(parent_state.path), n_valid=len(valid),
                    top_prob=float(row[int(top_idx[0])]),
                    H=float(-(_q * np.log(_q)).sum()) if _q.size else 0.0))
            if _LIB is not None and _in_library(parent_state):
                _liba = [_i for _i, (_o, _d) in enumerate(valid)
                         if (int(_o), tuple(int(a) + int(b)
                                            for a, b in zip(target, _d))) in _LIB]
                _brank = -1
                if _liba:
                    _ordL = np.argsort(-row)
                    _posL = {int(v): _r for _r, v in enumerate(_ordL)}
                    _brank = min(_posL[_i] for _i in _liba if _i in _posL)
                # how many of the ~20 CHILDREN this parent actually expands are
                # library actions. anyhit20 only guarantees >=1 of K, i.e. a
                # fraction >= 1/K; anything well above that is the model
                # beating its own guarantee.
                _tset = set(int(_i) for _i in top_idx)
                _nk = len(_tset & set(_liba))
                _LIB_PARENT.append(dict(
                    depth=len(parent_state.path), n_valid=len(valid),
                    avail=len(_liba), best_rank=_brank, K=int(K),
                    n_children=len(_tset), lib_children=_nk,
                    frac_children=round(_nk / max(len(_tset), 1), 3),
                    # CALIBRATION SHAPE. A state with `avail` equally-correct
                    # actions should carry ~1/avail on each, so top_prob*avail
                    # ~= 1 when calibrated and >>1 when the model has piled its
                    # mass on ONE member of a set it ought to be spreading over.
                    # lib_mass is the total probability on the closure set.
                    top_prob=float(row[int(top_idx[0])]),
                    lib_mass=float(sum(row[_i] for _i in _liba)),
                    conc=round(float(row[int(top_idx[0])]) * max(len(_liba), 1), 2),
                    in_topk=bool(0 <= _brank < K),
                    best_prob=(float(row[max(_liba, key=lambda i: row[i])])
                               if _liba else 0.0)))
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
            if _PROB_FLOOR > 0.0 and len(top_idx):
                _keep_i = [int(i) for i in top_idx if row[int(i)] >= _PROB_FLOOR]
                if not _keep_i:                 # never leave a state stranded
                    _keep_i = [int(top_idx[0])]
                top_idx = np.array(_keep_i, dtype=int)
            if _ENT_BONUS > 0.0 or _ENT_CUT > 0.0:
                try:
                    _n_expanded
                except NameError:
                    _n_expanded = 0
                    _n_ent_cut = 0
                _pr = np.asarray(row[:len(valid)], dtype=np.float64)
                _pr = _pr[_pr > 0]
                _parent_H = float(-(_pr * np.log(_pr)).sum()) if _pr.size else 0.0
                # AIM AT THE TAIL, NOT THE MEDIAN. The two populations overlap
                # heavily (median entropy 2.653 lib vs 2.089 non-lib at the
                # death window, a 1.3x difference), so no threshold separates
                # them. But ~7-8% of parents sit at top_prob>0.9, and those are
                # the ones whose children arrive at p~0.999 and take several
                # slots at once. A LOW cut (0.5-1.0; top_prob 0.999 gives
                # H~0.01) removes that tail while leaving the bulk untouched.
                if _ENT_CUT > 0.0 and _parent_H < _ENT_CUT and _n_expanded > 0:
                    _n_ent_cut += 1
                    continue        # overconfident parent: do not expand
                # counter lives ONLY on this path -- it is initialised in the
                # same block, so incrementing it outside would crash every run
                # that sets neither ENT_BONUS nor ENT_CUT, i.e. the DEFAULT.
                _n_expanded += 1
            for _rk, ai in enumerate(top_idx):
                ai = int(ai)
                op, delta = valid[ai]
                _t = time.time() if _v5_prof else 0
                # RANK SCORE (SAILIR_RANK_SCORE=1): feed the model's RANK rather
                # than its probability. top_idx is argsort(-row), so _rk is the
                # rank directly. Passing 1/(1+rank) makes the stored score
                # -log(1+rank): 0, -0.69, -1.10 for ranks 1,2,3.
                #
                # Why: the beam sorts on log p, but the model's mass is
                # saturated while its ranking is fine. Measured on 2,1,0 at the
                # two steps that lose the truth trajectory, the correct action
                # was ranked 3rd and 2nd -- inside every top-K metric we select
                # on -- yet carried p=0.062 and p=0.0104 against top picks of
                # 0.734 and 0.988. As log p that is a 1.0 and 4.6 nat deficit,
                # unreachable by any (w2, nm) reweighting. As rank it is 1.10
                # and 0.69 nats, which the structural terms can outvote.
                if _RANK_SCORE:
                    _ap = 1.0 / (1.0 + _rk)
                elif _REL_SCORE:
                    _mx = float(row[int(top_idx[0])])
                    _ap = float(row[ai]) / _mx if _mx > 0 else float(row[ai])
                elif _ENT_BONUS > 0.0:
                    _ap = float(row[ai]) * math.exp(_ENT_BONUS * _parent_H)
                else:
                    _ap = float(row[ai])
                result = apply_action_v5(
                    parent_state, target, op, delta, _ap,
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

        # EARLY SUCCESS (2026-08-04): if any freshly-generated child already
        # satisfies the success predicate, TAKE IT NOW — before the top-K prob
        # cull. A bucket-clearing move is correct regardless of the model's
        # probability for it; subjecting it to the probability ranking lets a
        # low-prob winner be generated and then discarded, which stalled the
        # frozen-7 runs for 2000 steps at nm=1 (the clearing move existed and
        # was reachable every step but never ranked in the top-40). This
        # generalizes the top-of-loop line-1632 check to the candidate set.
        # cand_metadata is parallel to candidates (appended together), so the
        # index maps directly; materialize the winner's lazy resolved_subs so
        # best_state._asdict() is consistent (the authoritative output is
        # replay_full_expr(best_state.path), which is path-only regardless).
        _win_i = next((i for i, c in enumerate(candidates)
                       if _is_success(c, target_sector)), None)
        if _win_i is not None:
            _wc = candidates[_win_i]
            _wp, _wt_tgt, _wsol = cand_metadata[_win_i]
            if lazy_rs:
                _wc = _materialize_lazy_rs(_wc, _wp, _wt_tgt, _wsol, start_w12)
            best_state = _wc
            if verbose:
                print(f'[v6 step {step}] EARLY SUCCESS — bucket-clearing move '
                      f'taken pre-cull (path_len={len(_wc.path)}'
                      + (f', lane={_wc.lane}' if beam_sort == 'dual' else '')
                      + ')', flush=True)
            break

        # Beam selection
        _t_sort = time.time() if _v5_prof else 0
        # Build id() → metadata-index map so we can find each survivor's parent.
        meta_by_id = {id(c): i for i, c in enumerate(candidates)}
        if beam_sort == 'weight':
            if _TOPN_PROB > 0:
                _byp = sorted(candidates, key=lambda c: -c.score)[:_TOPN_PROB]
                _res = {id(c) for c in _byp}
                _rest = [c for c in candidates if id(c) not in _res]
                _rest.sort(key=sort_weight)
                beam = _byp + _rest[:max(0, beam_width - len(_byp))]
            else:
                candidates.sort(key=sort_weight)
                beam = candidates[:beam_width]
        elif beam_sort == 'prob':
            if _W1_DROP and start_w12 is not None:
                _kept = [c for c in candidates
                         if c.max_w12[0] <= start_w12[0]]
                if _kept:            # never strand the beam
                    candidates = _kept
            candidates.sort(key=sort_prob)
            if _RESERVE_PARENT:
                # ONE RESERVED SLOT PER PARENT, then fill by probability.
                #
                # MEASURED JUSTIFICATION (soft_ep10, target 2,1,0, steps 49-51,
                # 18 library-consistent parents observed): a closure action was
                # enumerable for EVERY one of them (avail 1-48, never 0), and
                # the model ranked a closure action FIRST for 15 of 18 -- rank 0
                # out of 2400-3500 enumerated actions. The model is not
                # failing. But those rank-1-correct children carry p=0.05-0.11,
                # while non-library parents emit rank-1-wrong children at
                # p=0.999, so the global sort discards every correct one.
                #
                # score is log(action_prob) under the PARENT'S OWN softmax, over
                # action sets of wildly different size (70 to 3500) and
                # difficulty. Those numbers are not commensurable across
                # parents, yet the cull compares them directly.
                #
                # Why reservation and not re-scoring: normalising per parent
                # (SAILIR_REL_SCORE) scored 19 vs 52, and round-robin by rank
                # (SAILIR_RANK_SCORE) 42 -- both throw away the real signal
                # absolute probability carries about WHICH PARENTS deserve
                # expansion. Reservation keeps that signal for every slot beyond
                # the guarantee, and only guarantees each parent its own best
                # child. Parent identity is path[:-1]; candidates are already
                # sorted, so the first one seen per parent IS its best.
                _seen = set()
                _res, _rest = [], []
                for _c in candidates:
                    _k = tuple(_c.path[:-1])
                    if _k not in _seen and len(_res) < beam_width:
                        _seen.add(_k)
                        _res.append(_c)
                    else:
                        _rest.append(_c)
                beam = (_res + _rest)[:beam_width]
            else:
                beam = candidates[:beam_width]
        elif beam_sort == 'toptotal':
            # PER-PARENT POOL + MAX-LEX-WEIGHT RANKING.
            #
            #   1. from each beam state take only its top-N actions by model
            #      score (SAILIR_TOPN_PER_PARENT, default 3), so no single
            #      confident parent can flood the candidate pool;
            #   2. rank the pooled survivors by the MAX lex weight over the
            #      non-masters -- s.max_w12, the largest single remaining
            #      integral -- with the model score breaking ties.
            #
            # NOT total_w12. total_w12 is the SUM of the non-master weights,
            # which punishes the temporary EXPANSION that a correct reduction
            # has to pass through. Measured on 1,0,1,0,0,0,5,1,1,1 at depth 4:
            # the true child is TIED BEST on max weight ([10,1], 0 of 40 beam
            # states better) but WORST of all 40 on total weight ([50,5] vs a
            # beam spanning [20,2]..[40,4]) -- evicted one step after the model
            # ranked it #1 at p=0.848.
            #
            # NO n_non_masters term either: the tiebreak is the model score, as
            # specified. sort_weight carries (max_w12, n_non_masters, -score),
            # and that middle term is the same expansion penalty in another
            # form -- the true child above holds 5 non-masters against the
            # beam's 2-4, so an nm tiebreak demotes it for the very property
            # that makes it correct.
            _npp = int(os.environ.get('SAILIR_TOPN_PER_PARENT', '3') or 3)
            _by_parent = {}
            for _c in candidates:
                _mi = meta_by_id.get(id(_c))
                _par = cand_metadata[_mi][0] if _mi is not None else None
                _by_parent.setdefault(id(_par), []).append(_c)
            _kept = []
            for _grp in _by_parent.values():
                _grp.sort(key=lambda s: -s.score)     # model's own ranking
                _kept.extend(_grp[:_npp])
            # ascending max_w12: smaller weight is better, the codebase-wide
            # convention (same direction as sort_weight).
            # PRIMARY AND ONLY KEY: the MAX TOTAL LEX WEIGHT over the
            # non-masters. With SAILIR_BEAM_TOTAL=1 s.max_w12 is the FULL
            # ordering tuple (r, s, |a|) from _mw_tw_from_nm -- the single
            # source of truth -- not the truncated (r,s). Dropping |a| is
            # what made the key degenerate: measured, every one of the 40
            # beam states tied at [10,1], so the tiebreak became the whole
            # sort.
            #
            # NO model-probability tiebreak. A prob tiebreak fails exactly
            # where the model is unreliable, which is the case it exists to
            # handle. Python's sort is stable, so genuine ties keep pool
            # order (parent-grouped, model-ranked within a parent).
            _kept.sort(key=lambda s: s.max_w12)
            beam = _kept[:beam_width]
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
        elif beam_sort == 'wprob':
            # HETEROGENEOUS beam: half ranked by weight, half by the model's
            # local log-prob minus the nm drain penalty (SAILIR_NM_PENALTY = the
            # lambda). Unlike 'mixed', which splits between two WEIGHT rules
            # (max_w12 vs total_w12) and so shares their blind spots, this
            # splits between rules with DIFFERENT failure profiles: weight-sort
            # prunes weight-uphill stretches structurally and solved the three
            # previously-irreducible integrals, while prob-sort lets a confident
            # learned route survive those stretches and closed integral B in far
            # fewer steps. Neither dominates, and the wsort100 stragglers were
            # never tried under prob at all -- so run both halves at once rather
            # than betting the whole beam on one rule.
            #
            # Weight half FIRST, and it also backfills: if one rule produces too
            # few distinct states, the beam is topped up rather than left short,
            # and weight-sort is the half with the better solve record.
            half = beam_width // 2
            new_beam = []
            seen = set()

            def _take(ordered, upto):
                for c in ordered:
                    if len(new_beam) >= upto:
                        return
                    cid = id(c)
                    if cid not in seen:
                        seen.add(cid)
                        new_beam.append(c)

            by_w = sorted(candidates, key=sort_weight)
            _take(by_w, half)
            _take(sorted(candidates, key=sort_prob), beam_width)
            _take(by_w, beam_width)
            beam = new_beam
        elif beam_sort == 'dual':
            # TRULY DISJOINT DOUBLE BEAM. Two sub-beams that never compete:
            #   lane 0 -- ranked by sort_weight  (the original beam)
            #   lane 1 -- ranked by sort_prob    (model log-prob minus lambda*nm)
            # Each lane selects ONLY from candidates descended from itself
            # (children inherit `lane`), and each gets its own fixed
            # beam_width//2 slots with NO cross-lane backfill.
            #
            # This is the difference from 'wprob', which pools all candidates
            # and applies both sorts to the shared pool. There, a state only
            # prob-sort valued got one step of life and then had to out-compete
            # weight-favoured siblings to reproduce, so a prob lineage could be
            # starved; shortfalls were also backfilled from weight, never the
            # reverse. Measured on 1,4,0,1,1,0,0,0,2,1: pure prob-sort solved it
            # in 245 steps and the pooled wprob beam took 3,143 with the same
            # model -- i.e. pooling cost ~13x on that target. Here neither lane
            # can crowd the other out, so the run should be no worse than the
            # better of its two lanes.
            lane_w = max(1, beam_width // 2)
            lane0 = [c for c in candidates if c.lane == 0]
            lane1 = [c for c in candidates if c.lane == 1]
            beam = (sorted(lane0, key=sort_weight)[:lane_w]
                    + sorted(lane1, key=sort_prob)[:lane_w])
            if verbose:
                # Report BOTH lane populations. A lane persistently below
                # lane_w means it is starved of distinct candidates -- which
                # would silently turn this back into a single-lane search, the
                # exact failure this mode exists to prevent.
                print(f'[v6 step {step}] dual: lane0(w)={min(len(lane0), lane_w)}'
                      f'/{lane_w} from {len(lane0)} cand, '
                      f'lane1(p)={min(len(lane1), lane_w)}/{lane_w} from '
                      f'{len(lane1)} cand', flush=True)
        else:
            raise ValueError(f'unknown beam_sort {beam_sort}')

        # TRUTH-TRACE: record where the truth trajectory sits relative to this
        # step's selection. Placed AFTER the beam is chosen and BEFORE the
        # early macro-dedup, so `candidates` is the full generated set and
        # `beam` is exactly what survived the sort -- the two populations the
        # mode-A / mode-B distinction is defined over. No-op unless
        # SAILIR_TRUTH_TRACE is set.
        if _trace_fh is not None:
            _truth_trace_step(step, _trace_parents, candidates, beam,
                              sort_weight, sort_prob, cand_metadata)

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
        # KEY: expression AND the action SET (SAILIR_DEDUP_KEY).
        #   expr      legacy -- keys on the expression alone. UNSOUND: a state
        #             is (expr, subs), and two states with the same expression
        #             but different substitution stores have DIFFERENT available
        #             actions. Measured on 1,0,1,0,0,0,5,1,1,1: at step 3 this
        #             deleted the state on the winning path because a same-expr
        #             sibling scored 0.47 nats better (-2.4653 vs -2.9381) on
        #             the -score tiebreak; identical max_w12 and nm. The correct
        #             action was then unavailable at step 4 and the run drifted
        #             for 2,950 steps.
        #   exprpath  DEFAULT. Adds frozenset(path). resolved_subs is a
        #             deterministic function of the path, so the path
        #             distinguishes stores WITHOUT materialising them -- lazy
        #             states carry resolved_subs=None, so any key over the store
        #             itself would force the materialisation this early dedup
        #             exists to avoid. frozenset (not tuple) so two states that
        #             took the same actions in a different order still merge.
        _dedup_key = os.environ.get('SAILIR_DEDUP_KEY', 'exprpath')
        if len(beam) > 1:
            early_groups = {}
            for s in beam:
                efp = frozenset(s.expr.items())
                if _dedup_key == 'exprpath':
                    efp = (efp, frozenset(s.path))
                if beam_sort == 'dual':
                    efp = (efp, s.lane)   # keep the lanes separate (see above)
                cur = early_groups.get(efp)
                if cur is None or (s.max_w12, s.n_non_masters, -s.score) < \
                                  (cur.max_w12, cur.n_non_masters, -cur.score):
                    early_groups[efp] = s
            if len(early_groups) < len(beam):
                if verbose:
                    print(f'[v6 step {step}] early-dedup: {len(beam)} → '
                          f'{len(early_groups)} distinct expr (pre-materialize)',
                          flush=True)
                # EARLY-DEDUP TRUTH CHECK. The dedup at the TOP of the next step
                # carries a DEDUP_KILLED_TRUTH instrument, but this one runs
                # first and makes that one a no-op -- so a truth state killed
                # here was invisible. Records the survivor that displaced it and
                # both sort keys, because the key is (max_w12, nm, -score) on
                # states that share an EXPRESSION but not a substitution store.
                _newb = list(early_groups.values())
                if _trace_fh is not None and _truth_actions:
                    import json as _j4

                    def _de(st):
                        n = len(st.path)
                        return n if (n <= len(_truth_actions) and
                                     all(st.path[i] == _truth_actions[i]
                                         for i in range(n))) else -1
                    _pre_d = max((_de(x) for x in beam), default=-1)
                    _post_d = max((_de(x) for x in _newb), default=-1)
                    if _pre_d >= 0 and _post_d < _pre_d:
                        _victim = next(x for x in beam if _de(x) == _pre_d)
                        _efp = frozenset(_victim.expr.items())
                        if beam_sort == 'dual':
                            _efp = (_efp, _victim.lane)
                        _surv = early_groups.get(_efp)
                        _trace_fh.write(_j4.dumps(dict(
                            event='EARLY_DEDUP_KILLED_TRUTH', step=step,
                            depth=_pre_d, depth_after=_post_d,
                            n_before=len(beam), n_after=len(_newb),
                            victim_max_w12=list(_victim.max_w12[:2]),
                            victim_nm=int(_victim.n_non_masters),
                            victim_score=float(_victim.score),
                            # resolved_subs is None under LAZY_RS until
                            # materialisation -- report -1 rather than crash.
                            victim_n_subs=(len(_victim.resolved_subs)
                                           if _victim.resolved_subs is not None
                                           else -1),
                            surv_max_w12=(list(_surv.max_w12[:2])
                                          if _surv is not None else None),
                            surv_nm=(int(_surv.n_non_masters)
                                     if _surv is not None else None),
                            surv_score=(float(_surv.score)
                                        if _surv is not None else None),
                            surv_n_subs=(len(_surv.resolved_subs)
                                         if (_surv is not None
                                             and _surv.resolved_subs is not None)
                                         else -1),
                            same_expr=bool(_surv is not None))) + chr(10))
                        _trace_fh.flush()
                beam = _newb

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
            # p_top = the model's probability on the action that produced the
            # best state. Under SAILIR_SCORE=local, state.score IS the log-prob
            # of that single action, so exp() recovers it exactly. Under decay
            # it is a decayed sum and NOT a probability -- so only print it when
            # local, rather than emit a number that invites misreading.
            _ptop = (f'p_top={math.exp(max(best_in_beam.score, -60)):.4f} '
                     if _SCORE_MODE not in ('cumulative', 'decay') else '')
            print(f'[v6 step {step+1:>3}] beam={len(beam)} '
                  f'{_ptop}'
                  f'uniq_expr={n_uniq_expr} '
                  f'best mw={mw_best} nm={nm_best} '
                  f'rs={sz_rs} rs_vsz={rs_vsz} '
                  f'expr={len(best_in_beam.expr)} '
                  f'cand={len(candidates)} '
                  f'precap(tasks={_PRECAP_STATS[0]} tot={_PRECAP_STATS[1]} '
                  f'max={_PRECAP_STATS[2]} capped={_PRECAP_STATS[3]}) '
                  f't_step={time.time()-t_step:.1f}s '
                  f't_total={time.time()-t0:.1f}s',
                  flush=True)
            if _precap_dump_dir and _PRECAP_BUF:
                with open(f'{_precap_dump_dir}/precap_step{step}.pkl', 'wb') as _pf:
                    pickle.dump(_PRECAP_BUF, _pf)
            _PRECAP_BUF.clear()
            _PRECAP_STATS[:] = [0, 0, 0, 0]
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
                # THE REGISTRY. PackedEq stores integral IDs, and those IDs are
                # meaningful ONLY to the registry of the process that made
                # them. Without this table a resuming process builds an empty
                # registry and every id is out of range -- resume raised
                # IndexError in get_tuple() for any run with packed
                # substitutions, i.e. the production configuration.
                pickle.dump({'_registry': _V7_REGISTRY._to_tuple}, f)

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

    if _V9_CULL_C and _V9_CULL_C_FALLBACKS:
        # A silent fallback means the kernel did nothing for this run.
        print(f'  [cull-kernel] FELL BACK to Python '
              f'{_V9_CULL_C_FALLBACKS} times (radix did not fit)',
              flush=True)
    elif _V9_CULL_C:
        print('  [cull-kernel] active, 0 fallbacks', flush=True)
    if _V9_ORACLE_RANK and _V9_RANKS:
        import statistics as _stt
        _mr = [r[2] for r in _V9_RANKS if r[2] >= 0]
        _cr = [r[1] for r in _V9_RANKS if r[1] >= 0]
        print('[rank] ================ SUMMARY ================', flush=True)
        print(f'[rank] steps measured        : {len(_V9_RANKS)}', flush=True)
        if _cr:
            print(f'[rank] cull rank of truth   : median {_stt.median(_cr):.0f} '
                  f'min {min(_cr)} max {max(_cr)}', flush=True)
        if _mr:
            print(f'[rank] MODEL rank of truth  : median {_stt.median(_mr):.0f} '
                  f'min {min(_mr)} max {max(_mr)}', flush=True)
            for _k in (0, 1, 5, 20, 100):
                _n = sum(1 for r in _mr if r <= _k)
                print(f'[rank]   truth in model top-{_k + 1:<4}: '
                      f'{_n}/{len(_mr)} ({100.0 * _n / len(_mr):.0f}%)',
                      flush=True)
    if _V9_ORACLE is not None:
        _v9_oracle_report()
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
    # Build the class the checkpoint was TRAINED with. Defaults to the historical
    # IBPActionClassifier so existing checkpoints are unaffected; a 'nosubs'
    # checkpoint has no subs_enc.* keys and fails load_state_dict against the
    # full class.
    _variant = _cka.get('model_variant', 'full')
    # score_activation drives forward()'s SECOND return value, which this beam
    # consumes as action_prob. 'sigmoid' for bce-trained models -- softmax would
    # renormalise over the action set and reimpose the action-count dependence
    # those models are trained without.
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
    print(f'  active threshold      = (w1,w2) >= {start_w12}'
          + (f' AND total-order <= start ({_START_TOTAL_KEY})'
             if _STRIP_TOTAL else '  [total-order strip OFF]'), flush=True)
    print(f'  SINGLE-STEP reducer; success criterion = '
          f'{"TOTAL-weight active bucket drained" if _SUCCESS_TOTAL else "(w1,w2) active bucket drained"}',
          flush=True)
    # Strip sub-weight from the CACHED raws used by the search (cu built mod-
    # lower-weight). Must be BEFORE any raw is cached. Replay uses the unstripped
    # get_raw_equation, so final_expr is unaffected. SAILIR_STRIP_RAWS=0 disables.
    if os.environ.get('SAILIR_STRIP_RAWS', '1') != '0':
        # With the total-order strip on, the ACTIONS must be stripped the same
        # way as the expression, or the two disagree: the expression drops
        # same-(r,s)/larger-|abs| terms while every raw equation keeps
        # generating them. get_raw_equation already accepts a 3-tuple meaning
        # the FULL total ordering (keep unless below the start), and weight()
        # returns exactly (w0, w1, |abs|-tuple), so the start's own weight IS
        # the threshold. Out-of-cone terms stay exempt either way.
        _raw_thr = weight(start_int) if _STRIP_TOTAL else start_w12
        ibp_env.set_raw_strip_threshold(_raw_thr)
        if _SECTOR_RANK:
            # sector-senior order only (see onestep_worker_v7): out-of-cone
            # terms are exempt from the strip so the subsector filter can
            # reject sector-raising (den-row backward) actions. Legacy order
            # keeps the plain strip (out-of-cone sub-weight = genuinely below).
            ibp_env.set_raw_strip_cone(_sector_mask(start_int))
            print(f'  raw-strip: threshold {_raw_thr} '
                  f'({"TOTAL order" if _STRIP_TOTAL else "(w1,w2)"}), '
                  f'out-of-cone exempt', flush=True)
        else:
            print(f'  raw-strip: cached raws stripped to {_raw_thr} '
                  f'({"TOTAL order" if _STRIP_TOTAL else "(w1,w2)"})',
                  flush=True)

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
