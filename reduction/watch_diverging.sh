#!/bin/bash
# PERSISTENT diverging-worker watchdog. Runs detached, like the orchestrator.
#
# Every $INTERVAL seconds: find workers whose non-master count has exploded,
# record their full state, and condor_rm them. A diverging worker never
# finishes and orchestrator workers have NO wall cap, so each one holds a CPU
# forever.
#
# NO KILL/RESUBMIT LOOP IS POSSIBLE (verified in hierarchical_reduction.py):
# the reaper only acts when the output file exists, there is no missing-cluster
# resubmit path, and the straggler path -- the only resubmitter -- is disabled
# whenever --straggler-timeout > 1e8 (production sets 1e9). Each worker is
# killed exactly once and is remembered in killed_diverging.jsonl so it is
# never re-reported.
#
#   $1 = run dir      $2 = nm threshold (default 500)   $3 = interval s (600)
set -u
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/conda_envs/pyg4/bin/python
RUN=${1:?run dir required}
NM=${2:-500}
INTERVAL=${3:-600}
export PYTHONUNBUFFERED=1
echo "[watchdog] start $(date '+%F %T')  run=$RUN nm>=$NM every ${INTERVAL}s"
while true; do
    # Stop when the orchestrator for this run is gone -- nothing left to guard.
    if ! ps aux | grep -q "[h]ierarchical_reduction.*$(basename $RUN)"; then
        echo "[watchdog] orchestrator for $(basename $RUN) is gone -- exiting $(date '+%F %T')"
        break
    fi
    echo "--- $(date '+%F %T') ---"
    $PY $B/reduction/kill_diverging_workers.py "$RUN" --nm-min "$NM" 2>&1
    sleep "$INTERVAL"
done
