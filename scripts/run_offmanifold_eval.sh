#!/bin/bash
# Measure a checkpoint OFF MANIFOLD: roll it out on a fixed target set and count
# how often its own top-1 is a correct action at the states IT reaches.
#
#   $1  model checkpoint      $2  targets file      $3  tag      $4  workers
#
# Everything except the checkpoint is held fixed -- same targets, beam width,
# max-steps, max-actions, closure sources, mode -- so a difference is
# attributable to the model. The rollout is model-dependent, so the SET of
# states visited differs between arms; the comparable quantity is the RATE
#   already_right / (already_right + emitted)
# i.e. top-1 lands on a correct action, over labelable states.
set -uo pipefail
cd /home/shih/work/SAILIR_p101
MODEL=$1; TARGETS=$2; TAG=$3; W=${4:-8}
E=results/dagger/eval_$TAG
rm -rf "$E"; mkdir -p "$E" logs
cp "$TARGETS" "$E/targets.txt"
echo "=== off-manifold eval: $TAG ==="
echo "model  : $MODEL"
echo "targets: $(wc -l < "$E/targets.txt")  workers=$W"

split -n r/"$W" -d --additional-suffix=.txt "$E/targets.txt" "$E/shard_"
PIDS=()
for S in "$E"/shard_??.txt; do
  B=$(basename "$S" .txt)
  SAILIR_DAGGER_MODEL="$MODEL" \
  SAILIR_DAGGER_MISSING="$E/${B}_missing.txt" \
  SAILIR_DAGGER_LOG="$E/${B}.log" \
    nohup ./reduction/run_dagger_campaign.sh "$S" "$E/${B}_rows.jsonl" errors \
      > "$E/${B}_progress.log" 2>&1 &
  PIDS+=($!)
done
for P in "${PIDS[@]}"; do wait "$P"; done
cat "$E"/shard_*_rows.jsonl > "$E/rows.jsonl" 2>/dev/null
cat "$E"/shard_*_missing.txt 2>/dev/null | sort -u > "$E/missing_targets.txt"
echo "=== $TAG SUMMARY ==="
python -u scripts/dagger_round_stats.py "$E"
