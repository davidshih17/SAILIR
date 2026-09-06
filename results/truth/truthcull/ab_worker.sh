#!/bin/bash
# A/B on --iraws-keep-first, truth-closure minnew replay, one target per job.
#   $1 = integral (comma form)
#   $2 = K            (SAILIR_MAX_ACTIONS, the minnew cull)
#   $3 = keep-first   (50 = production default, 1000000 = effectively uncapped)
#
# WHY: --iraws-keep-first 50 stops any substitution past the 50th from ever
# becoming an indirect-action anchor, so from step 51 on the enumerated action
# list falls short of what the state supports. On gravity3L
# 2,0,0,1,0,1,0,1,1,1,0,0,0,0,0 that lost the reducing action at step 63 and the
# run died STUCK at 71; uncapped, the same worker follows the truth path exactly
# and succeeds. This measures whether that generalizes across all 60 targets.
# See results/truth/TRUTHCULL_DIVERGENCE.md.
#
# Capped baseline already on disk (out/K100, out/K1000, out/K10000000000):
# 0/60, 6/60, 6/60.
set -u
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
D=$B/results/truth/truthcull
export SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1
export SAILIR_ACTION_SELECT=truthminnew
export SAILIR_ACTION_SCORE=random   # truthminnew yields ONE action so this is
                                    # never consulted; sumseed returns NEGATIVE
                                    # scores and the decay score logs them.
# Raw-equation LRU cap REMOVED (1e9 = never evicts). The default 50,000 is
# smaller than |iraws| (~230/anchor) once the anchor cap is off, and a fixed-
# order scan over more keys than the cache holds misses on EVERY lookup --
# cyclic thrashing. Measured on node21, same target, both to step 400:
# t_step 21.69s -> 1.24s, t_total-to-300 655s -> 142s, peak RSS 1038MB -> 732MB.
# Pure memoization of (op,seed) -> raw equation, so results are UNCHANGED;
# only speed. See results/truth/RAW_EQ_CACHE_THRASH.md.
export SAILIR_RAW_EQ_CACHE_CAP=1000000000
export SAILIR_MAX_ACTIONS=$2
export SAILIR_TRUTHCULL_METRIC=${METRIC:-minnew}
# Complementary-criteria mode: split the K budget across these rankings and
# union the survivors, instead of spending it all on one total order.
[ -n "${UNION:-}" ] && export SAILIR_TRUTHCULL_UNION="$UNION"
# TAG MUST BE INJECTIVE. `tr -d '-'` silently maps distinct integrals to one
# tag once indices go negative: measured 1,377 collisions in an 18,000-target
# sample from the real reduction (90.2% of which carry negatives), e.g.
#   0,1,1,0,1,1,-1,3,1,1,...  and  0,1,1,0,1,1,1,3,-1,1,...
# both -> 0_1_1_0_1_1_1_3_1_1_... so they would share a closure file and
# overwrite each other's training rows. Harmless for the old dots-only corpus
# (no negatives => the strip was a no-op), which is why it survived.
# Dropping the strip leaves every EXISTING tag unchanged and matches
# eqact_dataset.tag_of(), which never stripped.
TAG=$(echo "$1" | tr ',' '_')
export SAILIR_TRUTH_CLOSURE=$D/../closures/out/${TAG}.json
# OUTSUF: distinguish runs that share a METRIC but not their conditions (the
# corrupted study is METRIC=upstream too, and must NOT overwrite the clean
# study's output). Empty by default, so every existing .sub is unaffected.
OUT=$D/out/${UNION:+u_}${UNION:-${METRIC:-minnew}}_kf$3_K$2${OUTSUF:-}
mkdir -p $OUT
# TRAINDIR: write this target's training rows where the caller can count them.
# ab_worker computes TAG; Condor cannot, so the path is derived here.
if [ -n "${TRAINDIR:-}" ]; then
  mkdir -p $TRAINDIR
  export SAILIR_TRUTHCULL_TRAIN=$TRAINDIR/${TAG}.jsonl
fi
PYTHONUNBUFFERED=1 /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
  $B/reduction/onestep_worker_truthcull.py \
  --topology $B/topology_input/gravity3L \
  --integral="$1" \
  --output $OUT/${TAG}.pkl \
  --model-checkpoint /dev/null \
  --iraws-keep-first $3 \
  --beam_width 1 --max_steps 20000 --prime 1009 --v7-cpus 1 2>&1
echo "EXIT=$?"
