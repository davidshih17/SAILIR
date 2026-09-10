#!/bin/bash
# Truth reductions for the L9 r=9 frontier integrals of g1023_greedy_certified.
#
# WHY THESE: the orchestrator's frontier shows L9r9:6 -- six top-level integrals
# the reduction still needs. THREE of them were killed by the 15GB memory guard
# and are never retried, so the campaign cannot finish without them:
#   (1, 0,1,1,1,1,1, 1,1,1,...)  r=9 s=0   KILLED (mem)
#   (1,-2,1,1,1,1,1, 1,1,1,...)  r=9 s=2   KILLED (mem)
#   (1, 1,1,1,1,1,1,-2,1,1,...)  r=9 s=2   KILLED (mem)
# and two more are still in flight.
#
# Recorded with truth_record_training_v2.py, the DATA-GEN-FAITHFUL recorder --
# not truth_record_variant.py, whose own docstring says do NOT use it for corpus
# generation. Faithfulness is the point: these samples must be in the same
# schema the model was TRAINED on, otherwise "how does the model rank the truth
# action" is measuring a format mismatch rather than the policy.
#
# Output feeds training/walk_truth_reduction.py, which walks each trajectory and
# reports, per step, where the correct action ranks among the legal ones, how
# many wrong actions outrank it, and how confident the model is.
set -u
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
OUT=$B/results/truth/l9r9
LOG=$OUT/logs

# Same total order as the workers. SECTOR_RANK is senior and must match, or the
# recorded target selection is not the one inference uses.
export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1
export PYTHONUNBUFFERED=1

mkdir -p $OUT $LOG

INTS=(
  "1,0,1,1,1,1,1,1,1,1,0,0,0,0,0"
  "1,-2,1,1,1,1,1,1,1,1,0,0,0,0,0"
  "1,1,1,1,1,1,1,-2,1,1,0,0,0,0,0"
  "1,1,1,1,1,1,1,0,1,1,0,0,0,0,0"
  "1,1,1,1,1,1,1,0,1,1,-2,0,0,0,0"
)

for I in "${INTS[@]}"; do
  TAG=$(echo "$I" | tr ',' '_' | tr -d ' ')
  O=$OUT/truth_$TAG.jsonl
  L=$LOG/truth_$TAG.log
  if [ -s "$O" ]; then echo "skip (exists): $TAG"; continue; fi
  rm -f "$L"                       # stale log misread as fresh output burns rounds
  echo "=== $I -> $O"
  $PY $B/reduction/truth_record_training_v2.py \
      --integral "$I" --output "$O" --dr 1 --ds 1 > "$L" 2>&1
  echo "    rc=$? lines=$(wc -l < "$O" 2>/dev/null || echo 0)"
done

echo "ALL TRUTH RECORDINGS DONE"
ls -la $OUT/*.jsonl 2>/dev/null | awk '{print "  "$NF" "$5" bytes"}'
