#!/bin/bash
# DRIP-FEED submitter: keep at most MAXQ of our jobs in the queue at a time.
#
# WHY THIS EXISTS. On 2026-09-04 a single 15,000-job submission (cluster
# 1919030) went in on top of ~16,000 jobs already run that day, while our tiers
# were also requesting 10 GB per core against the cluster's 4 GB/core limit.
# The schedd stopped responding ("SECMAN:2007: Read failure during security
# negotiation") and jobs could no longer be queried OR removed. The sysadmin had
# already warned that over-memory jobs get reaped and can wedge mount points on
# the batch nodes, which damages other users' jobs.
#
# Never submit the whole corpus at once again. This holds a bounded footprint and
# tops up only as jobs drain, so we never own the queue and a mistake costs one
# chunk instead of the campaign.
#
# Usage: drip_submit.sh <sub file> <targets file> [MAXQ] [CHUNK] [POLL_S]
set -u
SUB="$1"
TARGETS="$2"
MAXQ="${3:-2000}"      # never exceed this many of OUR jobs in the queue
CHUNK="${4:-1000}"     # submit at most this many at a time
POLL="${5:-120}"       # seconds between checks

D=$(dirname "$SUB")
WORK=$D/.drip
mkdir -p "$WORK"
split -l "$CHUNK" -d -a 4 --additional-suffix=.txt "$TARGETS" "$WORK/chunk_"
CHUNKS=($(ls "$WORK"/chunk_*.txt | sort))
echo "[drip] $(wc -l < "$TARGETS") targets -> ${#CHUNKS[@]} chunks of $CHUNK"
echo "[drip] cap: $MAXQ queued, poll ${POLL}s"

myq() {
    # Count OUR jobs. If the schedd cannot be reached, return a SENTINEL so the
    # loop WAITS rather than treating an unreachable schedd as an empty queue --
    # that would submit straight into a struggling scheduler.
    local out
    out=$(timeout 60 condor_q "$USER" -totals 2>/dev/null \
          | grep "Total for query" | grep -oE '^[^ ]*Total for query: [0-9]+' \
          | grep -oE '[0-9]+$')
    if [ -z "$out" ]; then echo "UNREACHABLE"; else echo "$out"; fi
}

i=0
while [ $i -lt ${#CHUNKS[@]} ]; do
    n=$(myq)
    if [ "$n" = "UNREACHABLE" ]; then
        echo "$(date +%H:%M:%S) [drip] schedd UNREACHABLE -- waiting, not submitting"
        sleep "$POLL"; continue
    fi
    # TRUE CEILING. The first version tested `n >= MAXQ`, which submits a full
    # chunk whenever the queue is one job under the cap -- real peak MAXQ+CHUNK.
    # Measured: MAXQ=6000 CHUNK=3000 peaked at 8,401 queued. Test whether the
    # chunk FITS, not whether we are already over.
    room=$(( MAXQ - n ))
    this=$(wc -l < "${CHUNKS[$i]}")
    if [ "$room" -lt "$this" ]; then
        echo "$(date +%H:%M:%S) [drip] queued=$n room=$room < chunk=$this -- waiting"
        sleep "$POLL"; continue
    fi
    c="${CHUNKS[$i]}"
    tmp="$WORK/current.sub"
    sed "s|^queue .* from .*|queue integral from $c|" "$SUB" > "$tmp"
    if timeout 300 condor_submit "$tmp" > "$WORK/last_submit.log" 2>&1; then
        echo "$(date +%H:%M:%S) [drip] chunk $((i+1))/${#CHUNKS[@]} ($(wc -l < "$c")) submitted; queued was $n"
        i=$((i+1))
    else
        echo "$(date +%H:%M:%S) [drip] submit FAILED, retrying after ${POLL}s:"
        tail -2 "$WORK/last_submit.log"
    fi
    sleep "$POLL"
done
echo "[drip] all ${#CHUNKS[@]} chunks submitted"
