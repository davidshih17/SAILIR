#!/usr/bin/env python
# =========================================================================
# per-step TARGETING selects the SINGLE highest integral in the workers'
# FULL total order (_target_key: sector rank when SAILIR_SECTOR_RANK=1,
# then (r,s), then |abs| lex), exactly like the training-data generator
# and the truth engine. v7 targeted ALL (r,s)-tied integrals per state
# (no lex tiebreak) — the long-flagged train/inference target-ordering
# mismatch ("Alignment is the next step"), measured 2026-08-02 to scatter
# =========================================================================
"""v6 beam search: macro-beam architecture on top of v5 strip-passenger.

Built from v5 (`greedy_reduce.py`). v5 wastes beam capacity because:
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
  - --beam-sort {weight,mixed,prob,dual,toptotal}
  - any() termination on first state with nm=0
"""
import argparse
import math
import operator
import json
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




_PICKLABLE_AUX_MARKER = '__v5_aux_v2__'






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












# ============================================================================
# v5 State
# ============================================================================

State_v5 = namedtuple(
    'State_v5',
    ['expr', 'resolved_subs',
     'score', 'path', 'n_non_masters',
     'max_w12', 'total_w12',  # cached sort keys, computed once at construction
     'aux_flat'],  # (cached_unique, union_bms, raw_id_to_idx, indirect_raws)
                   # or None to mean "rebuild from scratch on use"
)
State_v5.__new__.__defaults__ = (None, None, None)  # max_w12, total_w12, aux_flat


# ============================================================================
# Weight-active partition  —  SINGLE-STEP REDUCTION SEMANTICS  (READ THIS FIRST)
# ============================================================================
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
# path lengths -- verified via restart_offsets empty in 100/100 runs).
# ── TRUTH-TRACE ────────────────────────────────────────────────────────────
# SAILIR_TRUTH_TRACE=<recording.jsonl> follows the truth reduction alongside the
# search and records, step by step, whether the truth state is still inside the
#
# The question this answers cannot be answered from the ordinary logs. A run
# that ends up taking 3x the truth path length has left the truth trajectory
# right child or generated it and then culled it. Those need opposite fixes:
#
#             among the top-K expanded from it. The right move was never even a
#             candidate. Fix: larger or adaptive K.
#   MODE B -- the truth child WAS a candidate and lost the sort. Fix: the sort
#             rank and how far it missed the cutoff by, so the size of the fix
#             is measurable rather than guessed.
#
# the truth actions has exactly the truth expression, so equality of
# frozenset(expr.items()) at the same depth identifies it. Writing is
# append-only JSON lines and touches nothing the search reads, so behaviour is
# bit-identical with the variable unset -- and it is only worth paying for on a
# diagnostic run, since fingerprinting every candidate costs a pass over the
# candidate list.
# ---- CLOSURE-LIBRARY TRACE (SAILIR_LIB_TRACE=<lib json>) -----------------
#
# This replaces truth-PATH tracking, which was the wrong question twice over.
# (1) It followed the single LEX row, one arbitrary tie-break among ~17.9
#     holding a different, equally-reducing state. (2) It matched path
#     prefixes, and a state is (expr, resolved_subs) -- prefix matching
#     conflates distinct states and produced three retracted results.
#
# The library is a SET of (op, seed) actions -- the dependency closure of the
# target rule -- and consuming it in ANY order succeeds (first/last/random
# reach success in 338/352/352 steps). So membership is PATH-INDEPENDENT: a
# state is inside iff every action on its path came from the set. No matching,
# no tie-break, no conflation.


_LIB_PARENT = []          # per-step: one entry per library-consistent parent
# Every parent, tagged library or not, with AVAIL-FREE shape stats only:
# top_prob and entropy. `conc = top_prob*avail` is undefined for a non-library
# state (no correct set exists off the recorded reduction), so the comparable
# quantities are the raw concentration measures.
_ALL_PARENT = []




_truth_fp = []      # per truth step: frozenset(expr.items())
_frontier = -1      # furthest truth index the beam has ever held

# Each parent's row is a softmax over ITS OWN action set -- different sizes
# (70 to 2400 on one walk) and different difficulty -- so a parent where the
# model is sharp wins every slot from a parent where it is diffuse, regardless
# closure library: the BEST library candidate scored -2.245 (p=0.106) while
# non-library candidates ran to -0.001 (p=0.999), and 40 of the latter cleared
# the -2.531 cutoff. Meanwhile val_anyhit20 = 0.995-0.999, i.e. a correct
# action is essentially ALWAYS in the parent's own top-20. Both are true: rank
# WITHIN a parent is excellent, absolute probability ACROSS parents is not
# parents the model happens to be sharp about, and sharpness tracks how easy
# the state is, not whether the path is right.
#
# SAILIR_REL_SCORE=1 divides by the parent's own max, so the stored score
# becomes log(p_i / p_max): rank-1 from EVERY parent is 0, and within-parent
# gaps survive. Removes the cross-parent scale problem while keeping the
# within-state judgement anyhit20 says the model is good at.
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
# SAILIR_W1_DROP=1 -- FILTER form of the same separator: discard candidates
# whose w1 exceeds the START integral's w1, leaving `prob` as the PRIMARY sort.
#
# a secondary key -- measured 20 (ep10) and 31 (ep18) against baselines 52 and
# 60. Eleven variants in, the consistent pattern is that ANYTHING outranking
# the model's probability loses, because that ranking supplies the whole ~1.5x
# per-step enrichment holding r near 1. This drops provably-wrong candidates
# WITHOUT touching the ordering of what remains.
#
# Sound because the separation is categorical, not distributional: 0 of 1382
# library candidates had w1 > start_w1, against 917 of 1957 non-library.
# SAILIR_ENT_BONUS=lam adds lam * H(parent) in log space, i.e. favours children
# suspiciously peaked" rather than merely normalising every parent alike --
# worth separating because confidence is NOT always wrong (at step 50 the best
# library candidate carried p~1.0 and was correct), so a flat anti-confidence
# penalty would punish that too.
# SAILIR_ENT_CUT=x -- HARD cull: refuse to expand a parent whose action
# distribution has entropy below x, i.e. drop OVERCONFIDENT states outright.
# WARNING from the measured data: the library/non-library entropy ordering
# FLIPS with depth (steps 0-6 LIB 2.701 vs NON 2.844; steps 45-52 LIB 2.653 vs
# NON 2.089), so a fixed floor culls LIBRARY parents early and non-library ones
# late. Net effect unknown -- measure, do not argue.
# be emptied by the cut.









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
    if _START_TOTAL_KEY is not None:
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
    return True




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
# stripping by total order while success is still the (w1,w2) bucket test would
# declare success EARLY: the same-(r,s)-larger-|abs| terms would be dropped from
# the expression rather than eliminated, so n_non_masters could reach 0 with
# real work outstanding. Both workers force SAILIR_SUCCESS_TOTAL=1 before
# importing this module, so worker-driven runs get the strip on by default.


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
                            use_exprkeyed=True):
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
    return child._replace(aux_flat=new_aux)


# ============================================================================
# Model input with dummy subs (v5 variant)
# ============================================================================

def prepare_batched_input_v5_dummy(batch_data, device, max_subs=50,
                                    max_actions=1000):
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
# ============================================================================

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




# ===========================================================================
# Action-cap selection strategies (experiment; env SAILIR_ACTION_SELECT)
# ---------------------------------------------------------------------------
# When a (parent,target) task has > max_actions valid actions, only max_actions
# are fed to the model. Default 'first900' = valid[:max_actions] = the OLDEST
# actions (enumerate order is Phase-1a directs then Phase-1b indirects in iraws
# append/oldest->newest order); left as-is so prepare_batched_input still caps it
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
# full order. STALE, removed 2026-08-23: every target site is
# `tied = [min(nm, key=_target_key)]` (4 sites, each marked "v8: SINGLE
# full-order target"), and _target_key IS the training-data order (-r, -s,
# |abs|). The `tied` name is a leftover from the old behaviour.)
_METRIC_STRATEGIES = ('maxweight', 'shortest', 'sumweight')
# per-id metric arrays, cached + grown like packed_cu._bm_array (bounded memory)
_METRIC_CACHE = {'n': -1, 'twk': None, 'wt1': None, 'wt2': None,
                 'ism': None, 'sec': None}






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
# CLOSURE ROWS -- exactly the truthminnew availability condition
# (beam_search_truthcull.py: "keep only UNUSED CLOSURE rows -> take the best").
#
# The question it answers: when the model deviates from the truth path, can the
# expert still label the state? A task with zero available rows is one where
# truthminnew has NO action, so DAgger could not produce a label there.
# 'used' is per-state and reconstructed from that state's own path, since each
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
#
# SAILIR_DAGGER_MODE:
#   errors (default) -- emit ONLY where the model's top-1 is NOT a correct
#     action. truthminnew is WEAKER than the model where the model works (96-
#     the learner at the worse policy. Labelling only the model's mistakes
#     fixes recovery without overwriting good routing.
#   all -- emit every state (classic DAgger aggregation).
# SAILIR_DAGGER_DEADEND=1 -- also emit states where NO unused closure row is
#   legal, with an EMPTY valid_label_idxs. Those are evidenced dead ends (the
#   expert itself has no move), not merely unfamiliar states, so an all-zeros
#   BCE target there is a fact rather than an assumption.
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
_DAGGER_MISSING_SET = set()

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


# Cache keyed on the STORE OBJECT, not the step. _v9_cull runs once per
# (state, target) TASK, and _rs_as_dict materialises every PackedEq in the store
# until a substitution is added, and apply_action_v5 builds a NEW dict for the
# child, so identity is a sound key: a mutated store is a different object.
# the entry dies with the state.
_V9_SMW_CACHE = {}


# ── weight, without the waste ──────────────────────────────────────────────
# ibp_env.weight() makes THREE passes over the indices and allocates a 15-tuple
# of absolute values on every call. The cull uses only (w1, w2) and discards
# cumulative out of ~110s -- the whole of the step's unexplained "residual".
#
# _w12 does ONE pass, returns the two scalars, allocates nothing, and memoises
# on the integral (a pure function of it, and the same integrals recur heavily
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
_V9_ORACLE = None
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




_V9_RANK_PENDING = {}          # task index -> (depth, action, cull rank, space)
_V9_RANKS = []                 # (depth, cull_rank, model_rank, space, p, p_top)






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






# ── VALIDATED CONFIGURATION GUARD ──────────────────────────────────────────
# This build is bit-identical to the reference ONLY in the configuration it was
# certified in (125/125 targets, 15 phases). Every flag below CHANGES THE
# ALGORITHM, and the failure mode without this guard is silent: a different
# success test, ordering, or representation produces a different search with no
# error. Fourteen of the chain's env vars are set nowhere and rely on defaults
# being correct; this makes that dependency explicit and checked.
#
# Values are the ones OBSERVED in the certified runs, not source defaults:
#   worker  sets SUCCESS_TOTAL, PACKED_RS, STRIP_RAWS, END_OF_STEP_TRIM
#   wrapper sets SECTOR_RANK, NM_PENALTY, V9_UPENUM, V9_CULL
#   submit  sets BEAM_TOTAL
# THE certified configuration lives in reduction/greedy_certified.json, not
# here. It used to be duplicated: this dict, and a copy in
# hierarchical_reduction.py that pins the env written into every Condor submit.
# Two hand-maintained copies of one list is a silent-drift bug waiting to
# happen -- the orchestrator would pin one thing while this guard demanded
# another, and nothing would compare them.
#
# The record itself was CAPTURED from a certified run (greedy_worker.py stores
# os.environ in every result.pkl as 'env_snapshot'), not written from memory.
_CERT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'greedy_certified.json')
try:
    with open(_CERT_PATH) as _f:
        _CERT = json.load(_f)
except (OSError, ValueError) as _e:
    raise SystemExit(f'greedy: cannot read the certified record {_CERT_PATH}: '
                     f'{_e}\nThe worker refuses to run without it -- an absent '
                     f'record means nothing is pinned.')

_VALIDATED = {k: v for sec in ('env_set', 'env_model')
              for k, v in _CERT.get(sec, {}).items() if not k.startswith('_')}
_MUST_UNSET = _CERT.get('env_must_be_unset', {}).get('names', [])

# UNSET is a failure too, not a pass: an unset flag falls back to a source
# default that may differ from the certified value (SAILIR_SECTOR_RANK unset
# defaults to '0' = a DIFFERENT ordering). The caller must declare the
# configuration explicitly.
_bad = {k: os.environ.get(k) for k, v in _VALIDATED.items()
        if os.environ.get(k) != v}
# ...and the other half of the rule: a variable the certified run did NOT set.
# Checking only the names in the list cannot catch SAILIR_SYM_FIRST=1, which
# makes the worker return a symmetry rule with steps=0 and never run the search
# at all -- while every value above is still correct.
_leaked = {k: os.environ[k] for k in _MUST_UNSET if k in os.environ}
if _bad or _leaked:
    raise SystemExit(
        'greedy: refusing to run outside the validated configuration.\n'
        + '\n'.join(f'  {k}={v!r} but this build is certified only at '
                     f'{k}={_VALIDATED[k]!r}'
                     + (' (unset -> source default, which may differ)'
                        if v is None else '')
                     for k, v in sorted(_bad.items()))
        + ('\n' if _bad and _leaked else '')
        + '\n'.join(f'  {k}={v!r} but the certified run had it UNSET'
                     for k, v in sorted(_leaked.items()))
        + f'\nThese flags change the ALGORITHM, not just performance.'
        f'\nRecord: {_CERT_PATH}')


def greedy_reduce(env, model, start_expr, target_sector, start_w12,
                   max_steps=1000, device='cpu',
                   max_actions=1000, ckpt_path=None,
                   ckpt_every=50, verbose=True,
                   resume_from=None,
                   use_incremental_aux=True,
                   use_exprkeyed=True,
                   ckpt_every_step=False,
                   lazy_rs=True,
                   model_batch_chunk=8,
                   top_k=20,
                   nm_penalty=0.1):
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
    """
    if max_actions != 1000:
        raise SystemExit(
            f'greedy: max_actions={max_actions} but this build is validated\n'
            f'ONLY at the K=1000 cull that matches the training corpus. A\n'
            f'different cap feeds the model a different candidate list and\n'
            f'silently changes the search.')
    # nm_penalty and top_k define the SELECTION RULE itself:
    #   argmin over (-(log p - nm_penalty*nm), max_w12, nm)  across top_k children.
    # Either one moved is a different search, not a tuned one, and it would
    # fail silently -- every run still "succeeds", just along another path.
    if nm_penalty != 0.1:
        raise SystemExit(
            f'greedy: nm_penalty={nm_penalty} but this build is certified ONLY\n'
            f'at 0.1 (log p - 0.1*nm). The 125/125 run selected on this rule.')
    if top_k != 20:
        raise SystemExit(
            f'greedy: top_k={top_k} but this build is certified ONLY at 20.\n'
            f'top_k is the one-step lookahead width -- nm is observable only\n'
            f'AFTER an action is applied, so this is how many children get\n'
            f'built and scored before the argmin. It is not a beam.')

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
        _loaded = []
        # Convert saved aux from picklable (op, seed)-keyed form back to id()-
        # keyed in-memory form. This preserves the EXACT iraws structure the
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
            _loaded.append(State_v5(**s))
        if verbose:
            print(f'[v6 RESUME] loaded {len(_loaded)} states from step '
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
        state = initial

    best_state = state
    initial_mw = max_w12(start_expr, target_sector)
    t0 = time.time()

    # live State_v5 objects, so keeping `d` bound pins the ENTIRE serialized
    if resume_from is not None and os.path.exists(resume_from):
        del d

    # SOFT w2 (2026-08-15). sort_weight is lexicographic on
    # (max_w12, n_non_masters, -score), so max_w12 is absolute and the model is
    # unable to follow a truth path that goes temporarily UPHILL -- and the
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
        return s.score

    def sort_weight(s):
        return (s.max_w12, s.n_non_masters, -s.score)

    def sort_totalweight(s):
        return (s.total_w12, s.n_non_masters, -s.score)

    # Drain pressure for prob-sort (2026-08-03): pure local log-prob has no
    # and nm ratchets into the hundreds (frozen-7 divergence). Penalize each
    # active non-master by SAILIR_NM_PENALTY in log-prob units so states that
    # grow the bucket must buy it with genuinely better-ranked actions.
    # The PARAMETER, not the env var. SAILIR_NM_PENALTY stays in the import
    # guard so a caller that sets it expecting an effect gets an abort rather
    # than a silent no-op. Guard pins it to '0.1' == this default, so the
    # certified path is unchanged by construction.
    _nm_penalty = nm_penalty

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
        # being compared has the same path length and is divided by the same k.
        _sc = s.score
        return (-(_sc - _nm_penalty * s.n_non_masters),
                s.max_w12, s.n_non_masters)

    # Hoist memprobe config out of the per-step loop. Per-step we want at
    # most a set-membership check or an integer modulo — no env-var parsing.
    _memprobe_steps_set = None
    _memprobe_every = None
    _memprobe_start = None

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
        if os.environ.get('SAILIR_END_OF_STEP_TRIM', '1') != '0':
            _libc.malloc_trim.argtypes = [_ctypes.c_size_t]
            _libc.malloc_trim.restype = _ctypes.c_int
            _end_of_step_trim = _libc.malloc_trim
    except Exception:
        _end_of_step_trim = None

    # Gated by SAILIR_TRACE_BEAM_SETS=<path>. Per-step cost = one hash
    # over n_beam frozensets; per-step memory = 3 ints.


    # Focused runtime memory breakdown (SAILIR_MEM_BREAKDOWN=1). Each sampled
    # step we RESET VmHWM at step start (/proc/self/clear_refs) so the print's
    # peak_rss is THIS step's transient peak (not the all-time max), then report:
    #   peak_rss   = per-step transient peak (pre end-of-step trim)
    #   rss        = post-trim quiescent RSS
    #   smaps      = where RSS lives: heap (brk) / anon (mmap) / file
    #   glibc_transient = peak_rss - rss  (freed-then-trimmed each step)
    #   live_unaccounted = rss - sum_struct  (live, unnamed -> next target)
    # Every 4th sample a gc deep-walk-by-type names any live_unaccounted.








    # Diagnostic: dump per-task (expr_fp, target, n_v, picked_actions, V) every
    # picks and locate the first divergence. Off unless SAILIR_PICK_DUMP set.
    # v8 diagnostics: PRE-CAP action-space accounting. _PRECAP_STATS per step:
    # [n_tasks, total_precap, max_precap, n_capped_tasks]. SAILIR_PRECAP_DUMP
    # additionally dumps each task's FULL pre-cap list per step.
    _PRECAP_STATS = [0, 0, 0, 0]
    _PRECAP_BUF = []


    # The per-step enumerate phase (P1) is GIL-bound — dict membership, set/list
    # objects, so a child's read-only traversal dirties only small header pages,
    # Bit-identical to the serial loop: same three filters, same seed/delta, same
    # per-parent target order; results are re-sorted by parent_idx (stable) so the
    # model batch sees the identical task order. aux_flat is a pure derived cache
    # of resolved_subs, so a worker that rebuilds it returns the packed aux for the
    # main process to store (perf memo) — correctness is independent of it.




    for step in range(resume_step, max_steps):
        winning = state if _is_success(state, target_sector) else None
        if winning is not None:
            best_state = winning
            if verbose:
                print(f'[v6 step {step}] beam state drained — DONE '
                      f'(path_len={len(winning.path)})', flush=True)
            break

        # Reset the peak-RSS counter at the START of a sampled step so the
        # MEMBD print reads THIS step's transient peak (the candidate-gen
        # spike), not the all-time high-water mark.


        # trace jumps from "on the truth path" to "child never generated" with
        # nothing in between -- which is exactly where a depth-4 loss hid.


        # Build candidate list across all parents × valid actions
        candidates = []  # list of state_v5_child
        cand_metadata = []  # parallel list of (parent_state, target) for incremental aux
        all_state_summaries = []
        t_step = time.time()
        # V5_PROFILE=1: per-phase wall-clock instrumentation matching v4's
        # P1/P2/P3/P4 taxonomy from delta_beam_search.py.
        # P2 = model batched scoring: batch_prep + model_fwd
        # P4 = survivor materialize: attach_aux
        _p1 = {'nm_tied': 0.0, 'aux': 0.0, 'gv': 0.0}
        _p2 = {'batch_prep': 0.0, 'model_fwd': 0.0}
        _p3 = {'apply': 0.0, 'sort': 0.0}
        _p4 = {'attach_aux': 0.0}
        _ct = {'aux_build_calls': 0, 'aux_repack_calls': 0,
               'enum_calls': 0, 'apply_calls': 0, 'attach_calls': 0,
               'n_iraws_total': 0, 'n_valid_total': 0,
               'cu_size_max': 0, 'rs_max': 0}

        # Per-state: compute indirect cache, enumerate tied targets, score actions
        # Build tasks for model batched inference
        tasks = []  # (parent_idx, target, valid)
        parent_idx, s = 0, state
        nm = get_non_masters(s.expr, target_sector)
        tied = [min(nm, key=_target_key)]  # v8: SINGLE full-order target
        # Dummy subs (RS keys with empty values) — only used as keys for
        # enumerate_valid_actions iteration (it reads keys, not values).
        dummy_subs = {k: {} for k in s.resolved_subs.keys()}
        # Use cached aux_flat if available, else rebuild from scratch.
        if _V9_UPENUM:
            indirect_cache = None   # cu is never built
        for target in tied:
            if _V9_UPENUM:
                valid = _tc_upenum_valid(target, s, env)
            else:
                valid = _packed_cu.enumerate_valid_actions_with_indirect_cache_packed(
                    target, indirect_cache, s.resolved_subs,
                    env.ibp_t, env.li_t, env.shifts, 'subsector',
                    env._raw_eq_cache, _V7_REGISTRY, ibp_env.N_INDICES,
                )
            if valid:
                _PRECAP_STATS[0] += 1
                _PRECAP_STATS[1] += len(valid)
                _PRECAP_STATS[2] = max(_PRECAP_STATS[2], len(valid))
                if len(valid) > max_actions:
                    _PRECAP_STATS[3] += 1
                if _V9_CULL:
                    valid = _v9_cull(valid, target, s, env, max_actions)
                    if _V9_ORACLE is not None and _v9_on_truth_path(s):
                        # ONLY the state still on the recorded
                        # trajectory is comparable: it alone has the
                        # store the corpus was generated with. Verify
                        # its culled space, then force the oracle action
                        # so it stays on-path and later steps remain
                        # the model and are not checked.
                        _st = _V9_ORACLE_STATS
                        _st['deepest_on_path'] = max(
                            _st.get('deepest_on_path', 0), len(s.path))
                        _oa = _v9_oracle_check(len(s.path), target, valid)
                        if _oa is not None:
                            valid = [_oa]
                tasks.append((parent_idx, target, valid))

        # Instrumentation: dump tasks/valid lists at a specific step

        if not tasks:
            if not tasks:
                if verbose:
                    print(f'[v6 step {step}] no tasks — STUCK', flush=True)
                # All the diagnostics below only fire if STILL stuck after
                # the sol_fp fallback above (or if fallback was disabled).
                _real_stuck = True
            else:
                _real_stuck = False
            # Comprehensive forensic dump of the stuck state. Gated by env
            # var SAILIR_DUMP_STUCK=<path> so it only fires when requested.
            if _real_stuck:
                break

        # Run model batched on all tasks (optionally chunked).
        # Chunking bounds the peak transient activation memory in the forward
        # pass — large unchunked batches leave hundreds of MB in glibc's free
        # list every step (the activations are freed but glibc retains pages).
        # Per-chunk prepare + forward + write into a preallocated probs_all,
        # with explicit `del` so each chunk's tensors are released before the
        # next. None / 0 / >= len(batch_data) means no chunking (original).
        batch_data = []
        for parent_idx, target, valid in tasks:
            s = state
            batch_data.append((s.expr, s.resolved_subs, valid, target_sector, target))
        # Compute the global action width once so probs_all shape is stable.
        global_max_actions_eff = min(
            max(len(d[2]) for d in batch_data), max_actions)
        chunk_sz = model_batch_chunk if model_batch_chunk else len(batch_data)
        chunk_sz = max(1, min(chunk_sz, len(batch_data)))
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
            # corpus contains -- so these are resolved from the CURRENT store,
            # by the SAME EqResolver the training dataloader used. The expression
            # is truncated to SAILIR_EXPR_TERMS here for the same reason the
            # equations are capped at SAILIR_EQ_TERMS inside the resolver:
            # training saw both cut, and an uncut input is out of distribution.
            _eqkw = {}
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
        # OFF-MANIFOLD CONFIDENCE PROBE (SAILIR_CONF_PROBE=1). Top model score
        # cannot bias the sample. Training states all lie on a truth reduction
        # cull the corpus was generated with, so these scores are directly
        # comparable to on-manifold scores measured on val states.

        # For each task, take top-K actions by prob, apply, generate candidates
        K = max(1, top_k)
        _pick_dump_step = None
        for ti, (parent_idx, target, valid) in enumerate(tasks):
            _force_only = None      # set when the oracle forces this task
            n_v = min(len(valid), probs.shape[1])
            row = probs[ti, :n_v].numpy()
            parent_state = state
            top_idx = np.argsort(-row)[:K]
            _n_blocked = -1
            # ORACLE FORCING. One-hotting `row` above only makes the truth
            # action rank FIRST; top-K still expands K actions, so K-1
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
            # WAS A CLOSURE ACTION EVEN AVAILABLE HERE? For a parent that is
            # still inside the library, find every enumerated action that is a
            # library row for THIS target, and where the model ranked the best
            # of them. Distinguishes the two readings of a library death:
            #   avail=0            -> the state was structurally dead already;
            #                         the real mistake was culling diversity
            #   avail>0, rank>=K   -> closure actions existed and the model
            #                         ranked none inside the expanded top-K;
            #                         the fix is the model.
            # NOTE this also tests an assumption I had been making: library
            # membership does NOT guarantee a completable walk. The recorder
            # chooses its own target, so a library-consistent state can reach a
            # target none of its remaining rows act on.
            for _rk, ai in enumerate(top_idx):
                ai = int(ai)
                op, delta = valid[ai]
                # RANK SCORE (SAILIR_RANK_SCORE=1): feed the model's RANK rather
                # than its probability. top_idx is argsort(-row), so _rk is the
                # rank directly. Passing 1/(1+rank) makes the stored score
                # -log(1+rank): 0, -0.69, -1.10 for ranks 1,2,3.
                #
                # saturated while its ranking is fine. Measured on 2,1,0 at the
                # two steps that lose the truth trajectory, the correct action
                # was ranked 3rd and 2nd -- inside every top-K metric we select
                # on -- yet carried p=0.062 and p=0.0104 against top picks of
                # 0.734 and 0.988. As log p that is a 1.0 and 4.6 nat deficit,
                # unreachable by any (w2, nm) reweighting. As rank it is 1.10
                # and 0.69 nats, which the structural terms can outvote.
                _ap = float(row[ai])
                result = apply_action_v5(
                    parent_state, target, op, delta, _ap,
                    env, target_sector, start_w12,
                    use_incremental_aux=use_incremental_aux,
                    lazy_rs=lazy_rs,
                )
                if result is None:
                    continue
                child, sol = result
                candidates.append(child)
                cand_metadata.append((parent_state, target, sol))


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
                      + ')', flush=True)
            break

        # Build id() → metadata-index map so we can find each survivor's parent.
        meta_by_id = {id(c): i for i, c in enumerate(candidates)}
        candidates.sort(key=sort_prob)
        state = candidates[0]

        # TRUTH-TRACE: record where the truth trajectory sits relative to this
        # mode-A / mode-B distinction is defined over. No-op unless
        # SAILIR_TRUTH_TRACE is set.


        # then we've already paid the cost of materializing aux_flat for
        # every duplicate child that's about to be thrown away. Doing the
        # only on expr / max_w12 / n_non_masters / score — all of which
        # are unchanged by materialization) and saves the wasted work.
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
        #             exists to avoid. frozenset (not tuple) so two states that
        #             took the same actions in a different order still merge.


        # Post-selection survivors: materialize lazy_rs THEN attach aux in a
        # single pass so the original `c` is still in meta_by_id. Each survivor
        #
        # new raw equations that REGISTER new integrals in _V7_REGISTRY. In a
        # forked worker those registrations land in the child's COW registry; the
        # returned PackedEq states carry ids the MAIN registry never saw, so the
        # next step's enumerate hits reg.get_tuple(unknown_id) -> IndexError.
        # (P1 is immune because its results are registry-independent (op,delta)
        # bit-identical (FINAL REDUCTION IDENTICAL on (7,4)). BUT it is a MEASURED
        # NET LOSS and stays gated OFF: same-node A/B (node24, nw8+nthr8) was
        # SLOWER at every step (e.g. step29 97.1s on vs 69.4s off) and +13% peak
        # RSS (1877 vs 1659 MB). Cause: unlike P1's tiny (op,delta) outputs, each
        # P4 worker returns a full materialized state (resolved_subs + packed
        # aux_flat cu, ~1MB+), and pickling ~40 of those back + deserializing
        # serially in the main + the remap costs more than the 123s of attach_aux
        _t_aux = 0.0
        i, c = 0, state
        parent_state, target, sol = cand_metadata[meta_by_id[id(c)]]
        if lazy_rs:
            c = _materialize_lazy_rs(c, parent_state, target, sol, start_w12)
        if use_incremental_aux:
            c = _attach_incremental_aux(
                c, parent_state, target, env, target_sector,
                use_exprkeyed=use_exprkeyed,
            )
        state = c

        # Track best so far: use (max_w, n_non_masters) only — ignore score.
        # Score is cumulative negative log-prob, so the initial state always
        # has the "highest" score and would never be displaced by progress.
        def _progress_key(s):
            return (s.max_w12, s.n_non_masters,
                    -len(s.path))  # tiebreak: prefer longer path (more progress)

        best_in_beam = state
        if _progress_key(best_in_beam) < _progress_key(best_state):
            best_state = best_in_beam

        if verbose:
            mw_best = max_w12(best_in_beam.expr, target_sector)
            nm_best = best_in_beam.n_non_masters
            sz_rs = len(best_in_beam.resolved_subs)
            rs_vsz = sum(len(v) for v in best_in_beam.resolved_subs.values())
            n_uniq_expr = 1
            # p_top = the model's probability on the action that produced the
            # best state. Under SAILIR_SCORE=local, state.score IS the log-prob
            # of that single action, so exp() recovers it exactly. Under decay
            # it is a decayed sum and NOT a probability -- so only print it when
            # local, rather than emit a number that invites misreading.
            _ptop = f'p_top={math.exp(max(best_in_beam.score, -60)):.4f} '
            print(f'[v6 step {step+1:>3}] beam=1 '
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
            _PRECAP_BUF.clear()
            _PRECAP_STATS[:] = [0, 0, 0, 0]
        # Explicit aux dump (SAILIR_DUMP_AUX_STEPS="10,25"): print the best
        # state's aux_flat = (cu, ubm, rid, iraws) — lengths, the per-cu-entry
        # term counts, and samples — so depth-keyed vs exprkeyed aux can be
        # compared directly. SEPARATE block (not inside the profiling guard).

        def _stream_dump_ckpt(path):
            # to_dict-materialized cu is freed before the next. memray showed
            # simultaneously. Each state is an INDEPENDENT pickle frame (fresh
            # memo per pickle.dump) so `del sd` actually frees it; a single
            # shared Pickler would pin every object in its memo and save
            # nothing. Format: a meta dict (_streamed=True, n_states) then
            # n_states state-dicts in one file; the resume path detects
            # _streamed and reads n_states frames. Serialization-only change:
            # create fresh objects), so trajectory/result stay bit-identical.
            with open(path, 'wb') as f:
                pickle.dump({
                    '_streamed': True,
                    'step': step + 1,
                    'n_states': 1,
                    'target_sector': target_sector,
                    'start_w12': start_w12,
                }, f)
                for s in [state]:
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


        # Focused memory breakdown. peak_rss = this step's transient peak (VmHWM
        # was reset at step start); rss = post-trim quiescent. Named structures
        # use ONE shared seen-set so they partition (no double count). The two
        # gaps separate the dominant unaccounted into its two possible causes.

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
    return [state], best_state


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



if __name__ == '__main__':
    sys.exit(main())
