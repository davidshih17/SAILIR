#!/bin/bash
# MEASURE WHAT REGENERATION ACTUALLY BUYS.
#
#   ./scripts/run_regen_yield_test.sh [N_TARGETS] [COLLECT_JOBS] [REGEN_JOBS]
#
# Round 1 left 1,461 exhausted targets, ~83 CPU-h to regenerate. The one
# observation we have says that may buy nothing: re-collecting after the first
# 2 regenerations moved 5 states from `deadend` to `skipped_model_already_right`
# and emitted ZERO new rows -- the model had been right at every one, we simply
# had no closure row to certify it.
#
# The number that decides the 83 CPU-h is NEW ROWS PER REGENERATED CLOSURE.
# Three phases on the SAME small target set, everything else held fixed:
#
#   A  collect  -> rows_before, deadend_before, this set's exhausted targets
#   B  regen    -> build closures for exactly those exhausted targets
#   C  collect  -> rows_after, deadend_after   (identical command to A)
#
# The rollout is deterministic given the model, so any difference between A and
# C is attributable to the closure library alone.
set -uo pipefail
cd /home/shih/work/SAILIR_p101

N=${1:-20}
CJOBS=${2:-8}
RJOBS=${3:-6}
STAMP=$(date +%Y%m%d_%H%M%S)
Y=results/dagger/yield_$STAMP
mkdir -p "$Y" logs

R1=$(ls -td results/dagger/round_* | head -1)
echo "=== regen yield test $STAMP ==="
echo "Command: $0 $*"
echo "source round: $R1   targets=$N collect_jobs=$CJOBS regen_jobs=$RJOBS"

# Sample campaign targets from round 1 that we KNOW produced dead ends is not
# recorded per-target, so sample uniformly and let phase A measure it.
shuf -n "$N" --random-source=<(yes seed7) "$R1/targets.txt" > "$Y/targets.txt"
echo "sampled $(wc -l < "$Y/targets.txt") campaign targets"

collect () {            # $1 = phase tag
  local PH=$1
  mkdir -p "$Y/$PH"
  split -n r/"$CJOBS" -d --additional-suffix=.txt "$Y/targets.txt" "$Y/$PH/shard_"
  local PIDS=() S B
  for S in "$Y/$PH"/shard_??.txt; do
    B=$(basename "$S" .txt)
    SAILIR_DAGGER_MISSING="$Y/$PH/${B}_missing.txt" \
    SAILIR_DAGGER_LOG="$Y/$PH/${B}.log" \
      nohup ./reduction/run_dagger_campaign.sh "$S" "$Y/$PH/${B}_rows.jsonl" errors \
        > "$Y/$PH/${B}_progress.log" 2>&1 &
    PIDS+=($!)
  done
  local P; for P in "${PIDS[@]}"; do wait "$P"; done
  cat "$Y/$PH"/shard_*_rows.jsonl > "$Y/$PH/rows.jsonl" 2>/dev/null
  cat "$Y/$PH"/shard_*_missing.txt 2>/dev/null | sort -u > "$Y/$PH/missing.txt"
}

echo; echo "--- PHASE A: collect BEFORE ---"
collect before
echo "rows_before=$(wc -l < "$Y/before/rows.jsonl")  exhausted=$(wc -l < "$Y/before/missing.txt")"

echo; echo "--- PHASE B: regenerate those closures ---"
HAVE0=$(ls results/truth/closures/regen/out 2>/dev/null | wc -l)
SAILIR_REGEN_JOBS=$RJOBS ./reduction/run_closure_regen.sh "$Y/before/missing.txt"
HAVE1=$(ls results/truth/closures/regen/out 2>/dev/null | wc -l)
echo "regen closures: $HAVE0 -> $HAVE1  (+$((HAVE1-HAVE0)))"

echo; echo "--- PHASE C: collect AFTER (identical command) ---"
collect after
echo "rows_after=$(wc -l < "$Y/after/rows.jsonl")  exhausted=$(wc -l < "$Y/after/missing.txt")"

echo; echo "=== YIELD ==="
python -u scripts/regen_yield_report.py "$Y" "$((HAVE1-HAVE0))"
echo "run dir: $Y"
