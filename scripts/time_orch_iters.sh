#!/bin/bash
# Per-iteration wall time for a running orchestrator. The log records [Iter N]
# but NO timestamps, so iteration duration cannot be recovered from it after
# the fact -- this samples the log against the wall clock and writes a TSV.
set -u
LOG=$1; OUT=$2
last=""; mark=$(date +%s)
while true; do
  cur=$(grep -oP '\[Iter \K\d+' "$LOG" 2>/dev/null | tail -1)
  now=$(date +%s)
  if [ -n "$cur" ] && [ "$cur" != "$last" ]; then
    [ -n "$last" ] && echo -e "$(date +%H:%M:%S)\t$last\t$((now-mark))" >> "$OUT"
    last=$cur; mark=$now
  fi
  sleep 5
done
