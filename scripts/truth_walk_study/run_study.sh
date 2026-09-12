#!/bin/bash
# TRUTH-WALK MODEL SCORING STUDY -- single entry point.
#
# Full documentation, including every trap: docs/TRUTH_WALK_STUDY.md
#
# Answers: for these integrals, how does the model score the states it was
# TRAINED on, versus the states the beam wanders into? That separates a POLICY
# problem from a SEARCH problem, which no aggregate metric can do.
#
# This ORCHESTRATES the existing production scripts; it deliberately does not
# reimplement them, because the alignment guarantee comes from using the very
# same code that built the p=101 corpus:
#   results/truth/closures/make_groups.py         stage 1
#   results/truth/closures/batch_worker_p101.sh   stage 2   (SAILIR_PRIME=101)
#   results/truth/p101_cull_corpus/worker.sh      stage 3   (--prime 101, K=1000)
#   training/walk_truth_reduction.py              stage 5
#
# USAGE
#   run_study.sh <targets.txt> <study_name> [cap_mb]
#     targets.txt  one integral per line, comma-separated, e.g.
#                  1,0,1,1,1,1,1,1,1,1,0,0,0,0,0
#     study_name   output goes to results/truth/studies/<study_name>/
#
# STAGES 1-3 ARE THE EXPENSIVE ONES. Closure cost scales ~10x per dot level
# (measured on an L9 system: r<=10 1.5s, r<=11 20.7s, r<=12 172.4s), so budget
# hours for r>=13 -- the worker carries a 4h timeout and a 9 GB cap for exactly
# this reason. Run stages 1-3 on Condor for anything beyond a handful.
set -u
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

if [ $# -lt 2 ]; then sed -n '/^# USAGE/,/^#$/p' "$0"; exit 2; fi
TARGETS=$(readlink -f "$1")
NAME=$2
CAP=${3:-9000}
D=$B/results/truth/studies/$NAME

[ -s "$TARGETS" ] || { echo "REFUSING: $TARGETS missing or empty" >&2; exit 1; }
[ -e "$D" ] && { echo "REFUSING: $D exists (version the study name)" >&2; exit 1; }
mkdir -p $D/{groups,closures,logs,shard,train}   # train/ is written by stage 3

export SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1 PYTHONUNBUFFERED=1

echo "study    : $NAME"
echo "targets  : $(grep -c . "$TARGETS") integrals from $TARGETS"
echo "outdir   : $D"

# ── STAGE 1: group the targets by sector ────────────────────────────────────
# make_groups.py takes a PLAIN LIST and emits the per-sector JSON itself; the
# group schema {sector, rmax, smax, seeds, targets} is never hand-written.
# --no-merge keeps peak RSS ~2.3 GB instead of 12-24 GB.
echo "=== STAGE 1: groups ==="
$PY $B/results/truth/closures/make_groups.py \
    --targets "$TARGETS" --outdir $D/groups --budget 500000 --no-merge \
    > $D/logs/make_groups.log 2>&1
echo "  rc=$? groups=$(ls $D/groups/*.json 2>/dev/null | wc -l)"

# ── STAGE 2: closures AT PRIME 101 ──────────────────────────────────────────
# batch_worker_p101.sh, NEVER closures/worker.sh: the latter does not set
# SAILIR_PRIME and falls back to 1009. That is not a coefficient-only
# difference -- pivots are min(eq, key=tkey) over the surviving support, so a
# different prime masks a different term and picks a DIFFERENT pivot.
# Reuse first: the p=101 campaign left 12,584 closures in v4_p101_18k/out.
echo "=== STAGE 2: closures (SAILIR_PRIME=101) ==="
BATCH=$B/results/truth/closures/v4_p101_18k/out
n_reuse=0
while read -r I; do
  [ -z "$I" ] && continue
  TAG=$(echo "$I" | tr ',' '_')          # MUST NOT strip '-': see the doc
  if [ -s "$BATCH/${TAG}.json" ]; then
    cp "$BATCH/${TAG}.json" $D/closures/ && n_reuse=$((n_reuse+1))
  fi
done < "$TARGETS"
echo "  reused from the p101 batch: $n_reuse"
for G in $D/groups/*.json; do
  [ -e "$G" ] || continue
  echo "  building $(basename $G)"
  bash $B/results/truth/closures/batch_worker_p101.sh "$G" $D "$CAP" \
      >> $D/logs/closures.log 2>&1
  echo "    rc=$?"
done
cp -n $D/out/*.json $D/closures/ 2>/dev/null
echo "  closures now: $(ls $D/closures/*.json 2>/dev/null | wc -l)"

# ── STAGE 3: truth walk + training rows, by the CORPUS worker ───────────────
# The same worker that generated the p=101 corpus. This is the alignment
# guarantee: rows come from the corpus pipeline itself, so the ranking question
# measures the policy and not a schema mismatch. Do NOT substitute
# truth_record_training_v2.py -- it hardcodes P=1009 with no env override.
echo "=== STAGE 3: truth walks ==="
export SAILIR_CLOSURE_DIR=$D/closures
export SAILIR_WALK_OUTDIR=$D
export SAILIR_MEM_CAP_MB=7600           # cluster policy is 4 GB/core
while read -r I; do
  [ -z "$I" ] && continue
  TAG=$(echo "$I" | tr ',' '_')
  [ -s "$D/train/${TAG}.jsonl" ] && { echo "  skip (have rows): $TAG"; continue; }
  echo "  walking $I"
  bash $B/results/truth/p101_cull_corpus/worker.sh "$I" \
      > $D/logs/walk_${TAG}.log 2>&1
  echo "    rc=$? rows=$(wc -l < $D/train/${TAG}.jsonl 2>/dev/null || echo 0)"
done < "$TARGETS"

echo "=== STAGES 1-3 DONE ==="
echo "  train row files: $(ls $D/train/*.jsonl 2>/dev/null | wc -l)"
echo
echo "NEXT (stages 4-5), once the rows are complete:"
echo "  # 4. pack with --val-every 1 so EVERY step stays in val, in step order;"
echo "  #    walk_truth_reduction.py walks the LONGEST trajectory per target, so"
echo "  #    one trajectory per shard makes 'longest' unambiguous."
echo "  #    see results/truth/combine_and_pack.sh for the production packer"
echo "  # 5. score:"
echo "  $PY $B/training/walk_truth_reduction.py \\"
echo "      --topology $B/topology_input/gravity3L \\"
echo "      --shards_dir $D/shard \\"
echo "      --checkpoint p101=$B/checkpoints/gravity3L_p101_scratch/best_model.pt \\"
echo "      --n_scan_shards 1 --device cpu"
