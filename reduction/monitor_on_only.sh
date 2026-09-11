#!/bin/bash
# Watch the clean ON (8,4) full reduction to completion; append a 1-line snapshot
# every 120s; exit on "All done!" (or 14 h cap). On exit the harness re-invokes
# the agent, which then launches the cold OFF baseline for the fair comparison.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
ON=$BASE/results/probetop84_on
LOG=$BASE/results/probetop84_on_monitor.log
: > "$LOG"
START=$(date +%s); MAXSEC=$((14*3600))
while true; do
  now=$(date +%s); el=$(( (now-START)/60 ))
  last=$(grep "\[Iter" "$ON/logs/hierarchical.log" 2>/dev/null | tail -1)
  nw=$(ls "$ON"/work/results/*.pkl 2>/dev/null | wc -l)
  sym=$(grep -h "sym step" "$ON"/work/logs/*.out 2>/dev/null | wc -l)
  echo "t+${el}min $(date +%H:%M:%S) | results=$nw sym_lines=$sym | $last" >> "$LOG"
  if grep -q "All done!" "$ON/logs/hierarchical.log" 2>/dev/null; then
     echo "ON COMPLETE at t+${el}min : workers=$nw sym_lines=$sym" >> "$LOG"; break
  fi
  if [ $((now-START)) -gt $MAXSEC ]; then echo "TIMEOUT t+${el}min" >> "$LOG"; break; fi
  sleep 120
done
echo "MONITOR EXIT" >> "$LOG"