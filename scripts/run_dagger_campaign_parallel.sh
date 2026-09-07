#!/bin/bash
# Parallel DAgger collection round on THIS box.
#
#   ./scripts/run_dagger_campaign_parallel.sh [N_TARGETS] [N_WORKERS] [MODE]
#
# Shards the target list across workers, each running the EXISTING serial
# reduction/run_dagger_campaign.sh with its own output/missing/log files, then
# merges. Per-worker files are mandatory, not tidiness: a DAgger row carries a
# full expression and runs to many KB, far past PIPE_BUF, so concurrent appends
# to one file interleave mid-line and corrupt the jsonl.
#
# Targets are sampled from the shipped closure library, because a start target
# with no closure has no expert at all (run_dagger_collect.sh skips it). These
# are corpus targets, which is correct for DAgger: we are relabelling the
# MODEL's own state distribution on them, not inventing new problems.
#
# Concurrency: this is a SHARED box. Default 8 workers against 16 cores leaves
# headroom for other users and for a training job on GPU 0. Beam peak RSS was
# 749 MB/target, so 8 workers is ~6 GB -- RAM is not the constraint.
set -uo pipefail
cd /home/shih/work/SAILIR_p101

N=${1:-300}
W=${2:-8}
MODE=${3:-errors}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN=results/dagger/round_$STAMP
mkdir -p "$RUN" logs

echo "=== DAgger campaign $STAMP ==="
echo "Command: $0 $*"
echo "targets=$N workers=$W mode=$MODE  ->  $RUN"
echo "cores=$(nproc) load=$(cut -d' ' -f1-3 /proc/loadavg)"

# Deterministic sample of targets that HAVE a closure, EXCLUDING any target a
# previous round already walked. The rollout is deterministic given the model,
# so re-walking a collected target reproduces its rows exactly -- pure waste.
# (Once the model is retrained they become worth re-walking; that is a new
# DAgger iteration, and DONE_LIST should be cleared or pointed elsewhere.)
DONE=$RUN/.already_done
cat results/dagger/round_*/targets.txt 2>/dev/null | sort -u > "$DONE" || : > "$DONE"
ls results/truth/closures/v5_p101_59k/out \
  | sed 's/\.json$//' | tr '_' ',' | sort \
  | comm -23 - "$DONE" \
  | shuf -n "$N" --random-source=<(yes seed42) > "$RUN/targets.txt"
echo "sampled $(wc -l < "$RUN/targets.txt") targets "\
     "(excluded $(wc -l < "$DONE") already walked)"

# split into W shards
split -n r/"$W" -d --additional-suffix=.txt "$RUN/targets.txt" "$RUN/shard_"

PIDS=()
for S in "$RUN"/shard_*.txt; do
  B=$(basename "$S" .txt)
  SAILIR_DAGGER_MISSING="$RUN/${B}_missing.txt" \
  SAILIR_DAGGER_LOG="$RUN/${B}.log" \
    nohup ./reduction/run_dagger_campaign.sh "$S" "$RUN/${B}_rows.jsonl" "$MODE" \
      > "$RUN/${B}_progress.log" 2>&1 &
  PIDS+=($!)
done
echo "launched ${#PIDS[@]} workers: ${PIDS[*]}"
for P in "${PIDS[@]}"; do wait "$P"; done

# merge
cat "$RUN"/shard_*_rows.jsonl > "$RUN/rows.jsonl" 2>/dev/null
cat "$RUN"/shard_*_missing.txt 2>/dev/null | sort -u > "$RUN/missing_targets.txt"

echo
echo "=== ROUND SUMMARY ==="
echo "targets processed : $(cat "$RUN"/shard_??.txt | wc -l)"
echo "rows emitted      : $(wc -l < "$RUN/rows.jsonl")"
echo "exhausted targets : $(wc -l < "$RUN/missing_targets.txt")"
python -u scripts/dagger_round_stats.py "$RUN" 2>&1 | tail -20
echo "run dir: $RUN"
