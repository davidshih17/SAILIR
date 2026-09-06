#!/bin/bash
# p=101 K=1000 CORRUPTED CULL CORPUS -- production data generation.
#
# Follows results/truth/PRODUCTION_DATA_GENERATION.md exactly. Derived from
# results/truth/truthcull/ab_worker.sh (the script that generated
# ftcull1848_corrupt_shards, the corpus behind the first 125/125 model), with
# TWO parameter changes and nothing else:
#
#   1. --prime 101, and closures from the p=101 batch campaign. The truth
#      engine's row selection is arithmetic, so a p=1009 closure is not assumed
#      valid at p=101; these were generated at 101.
#   2. SAILIR_MEM_CAP_MB to hold the cluster's 4 GB/CORE policy (see README
#      1.3b: request_cpus = ceil(request_memory / 4GB)). A target over the
#      cap is DISCARDED to SAILIR_MEM_CAP_LIST, not failed -- it joins the
#      held-out hard set, which is scorable without ground truth because beam
#      success is self-verifying.
#
# ab_worker.sh is NOT reused directly: it hardcodes
# SAILIR_TRUTH_CLOSURE=$D/../closures/out/${TAG}.json, which is the p=1009
# closure tree. Editing it would silently change what every existing .sub in
# results/truth/truthcull/ does.
#
# DO NOT substitute reduction/truth_record_variant.py. That recorder does not
# rank or cull; it produces the pre-cull corr13 corpus and its action list is in
# enumeration order, so it cannot even be truncated into a culled one. That
# mistake cost a full 12,584-job campaign on 2026-09-03. See PITFALL 1.
#
#   $1 = integral, comma form
set -u
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
D=${SAILIR_WALK_OUTDIR:-$B/results/truth/p101_cull_corpus}
# Closure source and output dir are overridable so ONE worker serves every
# campaign. Defaults reproduce run 1 (the 18k closures) exactly, so an
# unqualified invocation behaves as before.
#   SAILIR_CLOSURE_DIR : where <tag>.json lives
#   SAILIR_WALK_OUTDIR : where out/, train/ and hard_memcap.txt are written
CLODIR=${SAILIR_CLOSURE_DIR:-$B/results/truth/closures/v4_p101_18k/out}

export SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1

# THE CULL. enumerate -> rank by upstream/fastmaxw -> cut to SAILIR_MAX_ACTIONS
# -> keep only UNUSED CLOSURE rows -> take the best. This is what makes the
# recorded valid_actions the SAME list beam_search_v9.py hands the model at
# inference, in the same rank order.
export SAILIR_ACTION_SELECT=truthminnew
export SAILIR_TRUTHCULL_METRIC=upstream
export SAILIR_MAX_ACTIONS=1000

# truthminnew yields exactly ONE action, so the score is never consulted;
# `sumseed` returns NEGATIVE scores and the decay score logs them.
export SAILIR_ACTION_SCORE=random

# Never evict. The default 50,000 is SMALLER than |iraws| (~230/anchor) once the
# anchor cap is off, so a fixed-order scan misses on EVERY lookup -- cyclic
# thrashing. Measured same target to step 400: t_step 21.69s -> 1.24s, peak RSS
# 1038MB -> 732MB. Results UNCHANGED; speed only.
export SAILIR_RAW_EQ_CACHE_CAP=1000000000

# CORRUPTION, exactly ftgen1848_corrupt.sub. Validated on the 60-target study
# (cluster 1912908): 57/60 solved, the SAME 3 failures as uncorrupted, ZERO
# StopIteration across 36,123 corrupted steps.
# The walk terminates only while FRAC * mean(N) < 1; here 0.2 * 3 = 0.6.
# Do not raise FRAC without redoing that arithmetic.
export SAILIR_CORRUPT_FRAC=0.2
export SAILIR_CORRUPT_MIN=2
export SAILIR_CORRUPT_MAX=4
export SAILIR_CORRUPT_SEED=12345

# Cap BELOW the Condor reservation so an outlier discards itself to the
# hard-target list instead of being killed by the OOM reaper. Default 7600
# pairs with an 8 GB / 2 CPU request.
export SAILIR_MEM_CAP_MB=${SAILIR_MEM_CAP_MB:-7600}
export SAILIR_MEM_CAP_LIST=$D/hard_memcap.txt

# TAG MUST BE INJECTIVE -- NO `tr -d '-'`.
# Deleting minus signs maps distinct integrals onto one tag once indices go
# negative: measured 1,024 collisions across these 12,584 targets (11,560
# distinct tags instead of 12,584), and 1,377 in the full 18,000 sample.
# Colliding targets share a closure file and overwrite each other's training
# rows. Harmless for the old dots-only corpora (no negatives, so the strip was a
# no-op) -- which is exactly why copying a dots-era worker reintroduces it.
# This matches eqact_dataset.tag_of(), which never stripped.
TAG=$(echo "$1" | tr ',' '_')

export SAILIR_TRUTH_CLOSURE=$CLODIR/${TAG}.json
if [ ! -f "$SAILIR_TRUTH_CLOSURE" ]; then
    echo "[worker] MISSING CLOSURE: $SAILIR_TRUTH_CLOSURE"
    exit 77
fi

OUT=$D/out
TRAINDIR=$D/train
mkdir -p "$OUT" "$TRAINDIR"
export SAILIR_TRUTHCULL_TRAIN=$TRAINDIR/${TAG}.jsonl

echo "[worker] target=$1 tag=$TAG closure=$(basename $SAILIR_TRUTH_CLOSURE)"

# WALL-CLOCK CAP, default 5h (SAILIR_WALK_WALL to override).
# Rows accrue roughly linearly with steps, but the RATE decays as the
# substitution store grows: measured on the two slowest run-2 targets, 5 and 2
# rows/min after 5h, with 362 and 1,174 rows still to go -- i.e. ~10h more for
# 0.07% of a 2.18M-row corpus. Not worth blocking the pack on.
# A capped target is exit 124 and belongs in the HARD-TARGET set (README 1.3c),
# and its PARTIAL train file must be deleted: a truncated walk is not the same
# as the contract-violation failures the reference corpus keeps.
timeout "${SAILIR_WALK_WALL:-18000}" \
PYTHONUNBUFFERED=1 /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
  $B/reduction/onestep_worker_truthcull.py \
  --topology $B/topology_input/gravity3L \
  --integral="$1" \
  --output $OUT/${TAG}.pkl \
  --model-checkpoint /dev/null \
  --iraws-keep-first 1000000 \
  --beam_width 1 --max_steps 20000 --prime 101 --v7-cpus 1 2>&1
rc=$?
# 124 = the timeout above. Delete the PARTIAL recording so a truncated walk can
# never reach the corpus; the target goes to the hard-target list instead.
if [ "$rc" = "124" ]; then
    echo "[worker] WALL-CLOCK CAP ${SAILIR_WALK_WALL:-18000}s -- discarding partial recording"
    rm -f "$SAILIR_TRUTHCULL_TRAIN"
    [ -n "${SAILIR_MEM_CAP_LIST:-}" ] && echo "$1	wallclock" >> "${SAILIR_MEM_CAP_LIST%.txt}_wall.txt"
fi
echo "[worker] EXIT=$rc rows=$(wc -l < $SAILIR_TRUTHCULL_TRAIN 2>/dev/null || echo 0)"
exit $rc
