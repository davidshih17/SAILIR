#!/bin/bash
# Exercise the exhaustion sink across several targets and produce a REAL
# regeneration work list. Targets are sampled from the shipped v5 closure
# library, so each has an expert; tags map back to integrals by '_'->','
# (minus signs contain no underscore, so this is lossless).
set -uo pipefail
cd /home/shih/work/SAILIR_p101
mkdir -p logs results/truth/closures/regen

N=${1:-8}
OUT=/tmp/claude-541005/-home-shih-work-RL-dilogs-claude/32ff41f5-704c-4867-9d31-c71582c75409/scratchpad/dagger_probe.jsonl
MISS=results/truth/closures/regen/missing_targets.txt
rm -f "$OUT" "$MISS"
: > logs/dagger_collect.log

ls results/truth/closures/v5_p101_59k/out | shuf -n "$N" --random-source=<(yes) \
  | sed 's/\.json$//' | tr '_' ',' > /tmp/dagger_probe_targets.txt
echo "targets:"; cat /tmp/dagger_probe_targets.txt

export SAILIR_DAGGER_MISSING="$MISS"
i=0
while read -r INT; do
  i=$((i+1))
  echo "=== [$i/$N] $INT ==="
  ./reduction/run_dagger_collect.sh "$INT" "$OUT" errors
  grep -E "\[DAGGER\]" logs/dagger_collect.log | tail -2
done < /tmp/dagger_probe_targets.txt

echo "=== TOTALS ==="
echo "rows emitted:        $(wc -l < "$OUT" 2>/dev/null || echo 0)"
echo "exhausted targets:   $(sort -u "$MISS" 2>/dev/null | wc -l)"
echo "--- work list head ---"; head -5 "$MISS" 2>/dev/null || echo "(none)"
