#!/bin/bash
# Rebuild closures for targets whose closure RAN OUT during DAgger collection.
#
#   ./reduction/run_closure_regen.sh [missing_targets.txt] [DR_USED] [DS_USED] [BUDGET] [CAP_MB] [WALL_S]
#
# TWO CAUSES OF EXHAUSTION, TWO REMEDIES
# -------------------------------------
# The work list is split automatically, because re-running the generator is
# only productive for one of them:
#
#   NEW  -- the target has NO closure anywhere in the shipped library. Going
#           off the truth path reaches targets the corpus never enumerated.
#           Measured on an 8-target probe: both exhausted targets were absent
#           from all 52,763 shipped closures. Generating at the SHIPPED
#           settings (dr=1 ds=1) produces a genuinely new closure here, so
#           these are built with DR_NEW/DS_NEW and stay interchangeable with
#           the library.
#
#   USED -- the target HAS a shipped closure, but every row of it legal at
#           this state has already been consumed along the path. Re-running is
#           pointless: truth_engine.worker_replay walks the ladder
#           [(dr,ds), (dr+1,ds), ...] and BREAKS at the first rung that solves
#           T, so for fixed (dr, ds, SAILIR_SEED_BUDGET) it is DETERMINISTIC
#           and would reproduce that same closure row for row. Only a bigger
#           seed box adds rows -- seeds(S, r+a, s+b) is a strict superset --
#           so these are rebuilt with the escalated DR_USED/DS_USED.
#
# --budget is the second knob for the USED arm: it is already 500000 in the
# shipped runs, but a deeper rung projects MORE seeds and any rung over the
# budget is `continue`-skipped, so too low a budget silently skips the very
# rungs being reached for.
#
# Cost is combinatorial in the box (make_groups.py: maximal boxes drove peak
# RSS to 12-24 GB), so the memory cap and wall-clock timeout from
# batch_worker_p101.sh are kept, and groups run ONE AT A TIME rather than
# fanning out. exit 42 = memory cap, 124 = timeout; both leave a marker in
# regen/retry/ and the loop continues to the next group.
#
# Output: results/truth/closures/regen/out/<tag>.json, the same schema as the
# shipped library. run_dagger_collect.sh already unions that directory in.
set -uo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn

MISSING=${1:-results/truth/closures/regen/missing_targets.txt}
DR_USED=${2:-2}          # escalated ladder start for targets that HAVE a closure
DS_USED=${3:-1}
BUDGET=${4:-500000}
CAP_MB=${5:-9000}
WALL_S=${6:-14400}
DR_NEW=1                 # shipped settings: batch_worker_p101.sh passes neither
DS_NEW=1

REGEN=results/truth/closures/regen
GRP=$REGEN/groups
mkdir -p "$GRP" "$REGEN/out" "$REGEN/retry" logs

# MUST match batch_worker_p101.sh or the regenerated closures are not
# interchangeable with the shipped ones: SAILIR_PRIME governs the whole
# elimination INCLUDING pivot choice (min(eq, key=tkey) over the surviving
# support), and SECTOR_RANK selects the canonical-masters convention.
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1
export SAILIR_KEEP_SYSTEMS=1 SAILIR_NO_SYSTEM_CACHE=1
export SAILIR_PRIME=101
export SAILIR_ROOT=/home/shih/work/SAILIR_p101

LOG=logs/closure_regen.log
{
  echo "=== closure regen $(date -Iseconds) ==="
  echo "Command: $0 $*"
  echo "missing=$MISSING budget=$BUDGET cap=${CAP_MB}MB wall=${WALL_S}s"
  echo "NEW arm dr=$DR_NEW ds=$DS_NEW   USED arm dr=$DR_USED ds=$DS_USED"
  echo "SAILIR_PRIME=$SAILIR_PRIME SAILIR_SECTOR_RANK=$SAILIR_SECTOR_RANK"
} | tee -a "$LOG"

[ -s "$MISSING" ] || { echo "no exhausted targets in $MISSING -- nothing to do" | tee -a "$LOG"; exit 0; }

# dedupe; drop targets already regenerated; split NEW vs USED
TODO_NEW=$REGEN/todo_new.txt
TODO_USED=$REGEN/todo_used.txt
: > "$TODO_NEW"; : > "$TODO_USED"
while read -r T; do
  [ -n "$T" ] || continue
  TAG=$(echo "$T" | tr ',' '_')          # minus signs PRESERVED (tag_of)
  [ -f "$REGEN/out/$TAG.json" ] && continue
  SHIPPED=""
  for D in results/truth/closures/v5_p101_59k/out results/truth/closures/v4_p101_18k/out \
           results/truth/closures/v5_p101_59k/retry results/truth/closures/v4_p101_18k/retry; do
    [ -f "$D/$TAG.json" ] && { SHIPPED="$D"; break; }
  done
  if [ -n "$SHIPPED" ]; then echo "$T" >> "$TODO_USED"; else echo "$T" >> "$TODO_NEW"; fi
done < <(sort -u "$MISSING")

N_NEW=$(wc -l < "$TODO_NEW"); N_USED=$(wc -l < "$TODO_USED")
echo "recorded=$(sort -u "$MISSING" | wc -l) new=$N_NEW used=$N_USED" | tee -a "$LOG"
[ "$N_NEW" -gt 0 ] || [ "$N_USED" -gt 0 ] || {
  echo "all already regenerated -- nothing to do" | tee -a "$LOG"; exit 0; }

run_arm() {                      # $1=todo  $2=dr  $3=ds  $4=label
  local TODO=$1 DR=$2 DS=$3 LABEL=$4
  [ -s "$TODO" ] || { echo "[$LABEL] no targets"; return 0; }
  local GDIR=$GRP/$LABEL
  mkdir -p "$GDIR"
  echo "######## ARM $LABEL  dr=$DR ds=$DS  targets=$(wc -l < "$TODO") ########"
  python -u results/truth/closures/make_groups.py \
      --targets "$TODO" --outdir "$GDIR" --budget "$BUDGET" --dr "$DR" --ds "$DS"
  shopt -s nullglob
  local GFILES=("$GDIR"/group_*.json)
  echo "[$LABEL] groups: ${#GFILES[@]}"
  local G rc
  for G in "${GFILES[@]}"; do
    echo "--- [$LABEL] $G ---"
    timeout "$WALL_S" python -u results/truth/closures/batch_closure.py \
        --group "$G" --outdir "$REGEN" \
        --dr "$DR" --ds "$DS" --budget "$BUDGET" --cap-mb "$CAP_MB" \
        --retry-dir "$REGEN/retry"
    rc=$?
    case $rc in
      0)   ;;
      42)  echo "[$LABEL] $(basename "$G") exit=42 MEMORY CAP";;
      124) echo "[$LABEL] $(basename "$G") exit=124 WALL-CLOCK TIMEOUT";;
      *)   echo "[$LABEL] $(basename "$G") exit=$rc";;
    esac
  done
}

{
  run_arm "$TODO_NEW"  "$DR_NEW"  "$DS_NEW"  new
  run_arm "$TODO_USED" "$DR_USED" "$DS_USED" used
  echo "regen closures on disk: $(ls "$REGEN/out" | wc -l)"
  echo "=== regen done $(date -Iseconds) ==="
} >> "$LOG" 2>&1

tail -12 "$LOG"
