#!/bin/bash
# Close the DAgger regeneration loop: re-run collection on the SAME target that
# exhausted, now that regenerated closures exist, and show the dead ends drop.
#
# Target 5 of the 8-target probe is the one that exhausted (5 dead ends across
# 2 distinct off-path targets). Everything else is held identical to that run --
# same model, beam width, max-steps, max-actions, mode -- so the only variable
# is the presence of results/truth/closures/regen/out.
set -uo pipefail
cd /home/shih/work/SAILIR_p101
mkdir -p logs

INT=0,2,1,0,1,2,2,-1,0,0,0,0,0,0,0
OUT=/tmp/claude-541005/-home-shih-work-RL-dilogs-claude/32ff41f5-704c-4867-9d31-c71582c75409/dagger_recollect.jsonl
MISS=results/truth/closures/regen/missing_after.txt
rm -f "$OUT" "$MISS"
: > logs/dagger_collect.log

echo "=== regen closures available: $(ls results/truth/closures/regen/out 2>/dev/null | wc -l) ==="
export SAILIR_DAGGER_MISSING="$MISS"
./reduction/run_dagger_collect.sh "$INT" "$OUT" errors

echo "=== BEFORE (from the 8-target probe) ==="
echo "  seen=19 emitted=10 skipped_model_already_right=4 deadend=5  -> 2 exhausted targets"
echo "=== AFTER ==="
grep -E "CLOSUREPROBE\] loaded|\[DAGGER\]" logs/dagger_collect.log
echo "  rows emitted:      $(wc -l < "$OUT" 2>/dev/null || echo 0)"
echo "  still exhausted:   $(sort -u "$MISS" 2>/dev/null | wc -l)"
