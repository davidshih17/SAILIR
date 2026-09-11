#!/bin/bash
# Kill the v22 g1023 orchestrator + ALL its condor workers (2026-07-31).
# Before killing: record every g1023 worker running >= 1 day into
# results/gr_reduce/g1023/longrunner_campaign_v2.txt with full stats.
# DOES NOT TOUCH: g885_symfirst workers, fid_ckpt producer, anything non-g1023.
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUT=$B/results/gr_reduce/g1023/longrunner_campaign_v2.txt
NOW=$(date +%s)

echo "# g1023 v22 workers >=1 day at kill time $(date '+%F %T')" > $OUT
echo "# jobid runtime_h memMB integral" >> $OUT

# snapshot ALL dshih jobs: id, start, memMB, args
condor_q dshih -af:j JobStartDate MemoryUsage Args > /tmp/v22_snapshot.txt
N_ALL=$(wc -l < /tmp/v22_snapshot.txt)
grep "g1023" /tmp/v22_snapshot.txt > /tmp/v22_jobs.txt
N_G1023=$(wc -l < /tmp/v22_jobs.txt)
echo "total dshih jobs: $N_ALL; g1023 workers: $N_G1023"

while read -r jid start mem rest; do
    [ "$start" = "undefined" ] && continue
    RT_H=$(( (NOW - start) / 3600 ))
    if [ $RT_H -ge 24 ]; then
        INTEG=$(echo "$rest" | grep -oE -- "--integral='[^']+'" | cut -d"'" -f2)
        echo "$jid ${RT_H}h ${mem}MB $INTEG" >> $OUT
    fi
done < /tmp/v22_jobs.txt
N_LONG=$(grep -vc "^#" $OUT)
echo "recorded $N_LONG longrunners (>=24h) -> $OUT"

# kill the orchestrator FIRST (so it cannot resubmit), by exact cmdline match
OP=$(pgrep -f "g1023/reduction.pkl")
if [ -n "$OP" ]; then
    ps -o pid,etime,cmd -p $OP | tail -1 | cut -c1-140
    kill $OP
    sleep 2
    if pgrep -f "g1023/reduction.pkl" > /dev/null; then
        kill -9 $(pgrep -f "g1023/reduction.pkl")
    fi
    echo "orchestrator killed (pid $OP)"
else
    echo "orchestrator not found"
fi

# remove exactly the g1023 worker jobs, by job id
cut -d' ' -f1 /tmp/v22_jobs.txt | while read -r jid; do
    condor_rm "$jid" > /dev/null 2>&1
done
echo "condor_rm issued for $N_G1023 g1023 jobs"
sleep 5
LEFT=$(condor_q dshih -af Args 2>/dev/null | grep -c "g1023")
echo "g1023 jobs still in queue: $LEFT"
echo "remaining dshih jobs (should be g885_symfirst + fid_ckpt):"
condor_q dshih -af Args 2>/dev/null | grep -oE "g885_symfirst|fid_ckpt|orch_bench" | sort | uniq -c
pgrep -f "g1023/reduction.pkl" > /dev/null && echo "WARNING: orch still alive" || echo "orchestrator confirmed dead"
