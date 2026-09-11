#!/bin/bash
# AUTO-DIAGNOSIS of slow orchestrator iterations (read-only, 2026-07-31).
# Watches the v22 log; when it goes silent >180s (mid-iteration), captures:
#   - 3 gdb stack snapshots 20s apart  (GC / dict-work / subprocess-wait)
#   - one 30s strace -c syscall profile (NFS stat storms, futex, pipes)
# Appends everything to results/gr_reduce/g1023/slow_iter_autodiag.log
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
LOG=$B/results/gr_reduce/g1023/logs/orch_v22.log
OUT=$B/results/gr_reduce/g1023/slow_iter_autodiag.log
while true; do
    OP=$(pgrep -f "g1023/reduction.pkl" | head -1)
    [ -z "$OP" ] && { echo "$(date '+%F %T') orchestrator gone" >> $OUT; sleep 60; continue; }
    AGE=$(( $(date +%s) - $(stat -c %Y $LOG) ))
    if [ $AGE -gt 180 ]; then
        echo "===== $(date '+%F %T') silent ${AGE}s — capturing (pid $OP)" >> $OUT
        for k in 1 2 3; do
            echo "--- gdb sample $k" >> $OUT
            timeout 15 gdb -p $OP -batch -ex "bt 6" 2>/dev/null \
                | grep -aE "#[0-5]" | head -6 >> $OUT
            sleep 20
        done
        echo "--- strace -c 30s" >> $OUT
        timeout 35 strace -c -p $OP 2>&1 | tail -25 >> $OUT
        echo "===== capture done; cooling down 300s" >> $OUT
        sleep 300
    else
        sleep 30
    fi
done
