#!/bin/bash
# Stop the g1023_greedy_certified orchestrator and ONLY its worker jobs.
#
# The queue also holds unrelated work of the user's (cosmology runs such as
# prospect_profile_H0_lcdm_*, and other clusters). A broad condor_rm would take
# those with it, so jobs are selected by BOTH greedy_worker.py AND this
# campaign's tag, and any cluster that mixes campaign and non-campaign jobs is
# removed per-job rather than per-cluster.
set -u
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
ORCH_PID=${ORCH_PID:?set ORCH_PID}
TAG=g1023_greedy_certified

echo "=== 1. verify the process we are about to kill ==="
if ! ps -p "$ORCH_PID" -o cmd --no-headers | grep -q "hierarchical_reduction.py"; then
    echo "REFUSING: pid $ORCH_PID is not a hierarchical_reduction.py process" >&2
    ps -p "$ORCH_PID" -o pid,cmd --no-headers >&2 || true
    exit 1
fi
ps -p "$ORCH_PID" -o pid,etime --no-headers | sed 's/^/  orchestrator: /'

echo "=== 2. select jobs: greedy_worker.py AND tag, with a mixed-cluster check ==="
$PY "$B/reduction/select_campaign_jobs.py" "$TAG" > /tmp/campaign_jobs.txt || exit 1
sed 's/^/  /' /tmp/campaign_jobs.txt
IDS=$(grep -v '^#' /tmp/campaign_jobs.txt | tr '\n' ' ')
if [ -z "${IDS// /}" ]; then
    echo "  no campaign jobs in the queue"
fi

echo "=== 3. stop the orchestrator FIRST so it cannot resubmit ==="
kill "$ORCH_PID"
sleep 5
if ps -p "$ORCH_PID" >/dev/null 2>&1; then
    echo "  still alive after TERM, sending KILL"
    kill -9 "$ORCH_PID"; sleep 3
fi
ps -p "$ORCH_PID" >/dev/null 2>&1 && echo "  *** STILL ALIVE ***" || echo "  orchestrator stopped"

echo "=== 4. remove its worker jobs ==="
if [ -n "${IDS// /}" ]; then
    # shellcheck disable=SC2086
    condor_rm $IDS 2>&1 | tail -3 | sed 's/^/  /'
fi

echo "=== 5. verify: campaign gone, everything else untouched ==="
sleep 5
$PY "$B/reduction/select_campaign_jobs.py" "$TAG" --count-only | sed 's/^/  /'
echo "  --- non-campaign jobs still in the queue (these MUST survive) ---"
condor_q -af ClusterId Args 2>/dev/null | grep -vc "greedy_worker.py" | sed 's/^/  surviving non-worker jobs: /'
condor_q -totals 2>/dev/null | tail -1 | sed 's/^/  /'
