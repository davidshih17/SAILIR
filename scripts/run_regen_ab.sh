#!/bin/bash
# A/B the 32 ALREADY-BUILT regenerated closures under the FIXED scoped probe.
#
# The earlier yield test is void: it was measured with the pooled probe, which
# widened the oracle 36x and manufactured its own result. This re-measures the
# same question with no closure building at all -- the closures already exist,
# so the only variable is whether the collector can see them.
#
#   arm A  regen dir hidden (empty dir)  -> baseline
#   arm B  regen dir visible             -> +32 targets' closures, scoped
set -uo pipefail
cd /home/shih/work/SAILIR_p101
EMPTY=/tmp/claude-541005/-home-shih-work-RL-dilogs-claude/32ff41f5-704c-4867-9d31-c71582c75409/scratchpad/empty_regen
STAMP=$(date +%Y%m%d_%H%M%S); AB=results/dagger/ab_$STAMP
mkdir -p "$AB" "$EMPTY"
# same 20 targets the earlier yield test used
Y=$(ls -td results/dagger/yield_* | head -1)
cp "$Y/targets.txt" "$AB/targets.txt"
echo "=== regen A/B $STAMP === targets=$(wc -l < "$AB/targets.txt")  closures available=$(ls results/truth/closures/regen/out | wc -l)"

run_arm () {   # $1=name  $2=regen dir
  local PH=$1 RD=$2
  mkdir -p "$AB/$PH"
  split -n r/8 -d --additional-suffix=.txt "$AB/targets.txt" "$AB/$PH/shard_"
  local PIDS=() S B
  for S in "$AB/$PH"/shard_??.txt; do
    B=$(basename "$S" .txt)
    SAILIR_REGEN_DIR="$RD" \
    SAILIR_DAGGER_MISSING="$AB/$PH/${B}_missing.txt" \
    SAILIR_DAGGER_LOG="$AB/$PH/${B}.log" \
      nohup ./reduction/run_dagger_campaign.sh "$S" "$AB/$PH/${B}_rows.jsonl" errors \
        > "$AB/$PH/${B}_progress.log" 2>&1 &
    PIDS+=($!)
  done
  local P; for P in "${PIDS[@]}"; do wait "$P"; done
  cat "$AB/$PH"/shard_*_rows.jsonl > "$AB/$PH/rows.jsonl" 2>/dev/null
  cat "$AB/$PH"/shard_*_missing.txt 2>/dev/null | sort -u > "$AB/$PH/missing.txt"
  echo "[$PH] rows=$(wc -l < "$AB/$PH/rows.jsonl") exhausted=$(wc -l < "$AB/$PH/missing.txt")"
}

echo "--- ARM A: regen hidden ---"; run_arm before "$EMPTY"
echo "--- ARM B: regen visible ---"; run_arm after results/truth/closures/regen/out
echo; echo "=== RESULT ==="
python -u scripts/regen_yield_report.py "$AB" "$(ls results/truth/closures/regen/out | wc -l)"
echo "run dir: $AB"
