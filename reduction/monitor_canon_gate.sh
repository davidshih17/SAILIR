#!/bin/bash
set -u
R=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
LOG=$R/results/canon_gate.log; : > "$LOG"
PY=$R/../RL_MIR_IBP/conda_env/bin/python
for i in $(seq 1 360); do
  if grep -q "All done!" $R/results/canon_on/logs/hierarchical.log 2>/dev/null; then
     echo "canon_on COMPLETE (poll $i)" >> "$LOG"; break; fi
  sleep 20
done
echo "=== last iter ===" >> "$LOG"
grep '\[Iter' $R/results/canon_on/logs/hierarchical.log | tail -1 >> "$LOG"
echo "=== worker crashes? ===" >> "$LOG"
grep -rl Traceback $R/results/canon_on/work/logs/*.err 2>/dev/null | wc -l >> "$LOG"
echo "=== GATE ===" >> "$LOG"
$PY $R/reduction/canon_gate.py >> "$LOG" 2>&1
echo "MONITOR EXIT" >> "$LOG"
