#!/bin/bash
# Background monitor for the (8,4) probe-top OFF vs ON full reductions.
# Appends a 2-line snapshot every 120s; exits when BOTH print "All done!"
# (or after a 14 h safety cap), then runs the correctness/worker-count compare.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OFF=$BASE/results/probetop84_off
ON=$BASE/results/probetop84_on
LOG=$BASE/results/probetop84_monitor.log
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

snap () {
  local d=$1
  local last; last=$(grep "\[Iter" "$d/logs/hierarchical.log" 2>/dev/null | tail -1)
  local nw;   nw=$(ls "$d"/work/results/*.pkl 2>/dev/null | wc -l)
  local sym;  sym=$(grep -h "sym step" "$d"/work/logs/*.out 2>/dev/null | wc -l)
  echo "    results=$nw sym_lines=$sym | $last"
}
done_p () { grep -q "All done!" "$1/logs/hierarchical.log" 2>/dev/null; }

: > "$LOG"
START=$(date +%s); MAXSEC=$((14*3600))
while true; do
  now=$(date +%s); el=$(( (now-START)/60 ))
  { echo "===== t+${el}min $(date +%H:%M:%S) ====="
    echo "  OFF:"; snap "$OFF"
    echo "  ON :"; snap "$ON"; } >> "$LOG"
  if done_p "$OFF" && done_p "$ON"; then echo "BOTH COMPLETE at t+${el}min" >> "$LOG"; break; fi
  if [ $((now-START)) -gt $MAXSEC ]; then echo "TIMEOUT at t+${el}min" >> "$LOG"; break; fi
  sleep 120
done
echo "===== FINAL COMPARE =====" >> "$LOG"
$PY "$BASE/reduction/compare_symgate.py" "$OFF" "$ON" >> "$LOG" 2>&1
echo "MONITOR EXIT" >> "$LOG"