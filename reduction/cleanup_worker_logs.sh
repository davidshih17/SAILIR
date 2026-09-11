#!/bin/bash
# Periodic worker-log cleanup for a hierarchical_reduction work dir.
# Archives (tar, then deletes) the .out/.err/.log files of workers whose
# SUCCESSFUL result pkl is already banked in results/, plus consumed batch
# .sub files — keeping logs of failed / in-flight workers and anything
# modified in the last 60 minutes. results/ (the cache) is never touched.
#
# Usage: cleanup_worker_logs.sh <work_dir>     e.g. results/gr_reduce/g1023/work
# Archive + report land next to the work dir (../logs_archive_<date>.tar, appended).
set -e
WORK=$(readlink -f "$1")
[ -d "$WORK/logs" ] || { echo "no logs/ under $WORK"; exit 1; }
ARCH=$(dirname "$WORK")/worker_logs_archive_$(date +%Y%m%d).tar
LIST=$(mktemp)

cd "$WORK"
# candidate log files: older than 60 min, with a banked result pkl
find logs -maxdepth 1 -type f -mmin +60 \( -name 'async_*.out' -o -name 'async_*.err' -o -name 'async_*.log' \) \
  | while read -r f; do
      base=$(basename "$f"); base=${base%.*}
      [ -f "results/$base.pkl" ] && echo "$f"
    done > "$LIST"
# consumed batch submit files older than 60 min
find . -maxdepth 1 -type f -mmin +60 -name 'batch_async_*.sub' >> "$LIST"

N=$(wc -l < "$LIST")
if [ "$N" -eq 0 ]; then
    echo "nothing to clean in $WORK"; rm -f "$LIST"; exit 0
fi
# append to (or create) today's archive, then delete the archived files
if [ -f "$ARCH" ]; then
    tar -rf "$ARCH" -T "$LIST"
else
    tar -cf "$ARCH" -T "$LIST"
fi
xargs -a "$LIST" rm -f
rm -f "$LIST"
echo "$(date '+%F %T') archived+removed $N files from $WORK -> $ARCH"
echo "remaining in logs/: $(ls logs | wc -l)"
