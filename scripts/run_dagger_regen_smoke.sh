#!/bin/bash
# Smoke-test the closure-regeneration wiring on ONE target that HAS a shipped
# closure. Validates, without needing the missing gravity3L engine pkls:
#   - the ':'-separated union loader (regen dir + shipped file)
#   - SAILIR_DAGGER_MISSING records exhausted targets
#   - collection itself did not regress
set -uo pipefail
cd /home/shih/work/SAILIR_p101
mkdir -p logs results/truth/closures/regen/out

INT=1,0,2,0,1,2,2,0,1,1,0,0,0,0,-1
OUT=/tmp/claude-541005/-home-shih-work-RL-dilogs-claude/32ff41f5-704c-4867-9d31-c71582c75409/scratchpad/dagger_smoke.jsonl
MISS=results/truth/closures/regen/missing_smoke.txt
rm -f "$OUT" "$MISS" logs/dagger_collect.log

export SAILIR_DAGGER_MISSING="$MISS"
./reduction/run_dagger_collect.sh "$INT" "$OUT" errors

echo "=== CLOSUREPROBE / DAGGER lines ==="
grep -E "CLOSUREPROBE|DAGGER" logs/dagger_collect.log || echo "(none)"
echo "=== rows emitted ==="
wc -l < "$OUT" 2>/dev/null || echo 0
echo "=== exhausted targets recorded ==="
wc -l < "$MISS" 2>/dev/null || echo 0
head -3 "$MISS" 2>/dev/null
