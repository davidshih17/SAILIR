#!/bin/bash
# DAgger collection: roll out the MODEL on a target, label the states IT visits
# with the corpus's own expert (unused closure rows), emit jsonl training rows.
#
#   $1  integral (comma-separated)
#   $2  output jsonl (appended)
#   $3  mode: errors (default) | all
#
# Closure sources are UNIONed, regenerated ones FIRST:
#   results/truth/closures/regen/out       <- built by run_closure_regen.sh
#   results/truth/closures/v5_p101_59k/... <- the shipped library
# Targets whose closure runs out are appended to $MISSING; feed that file to
# run_closure_regen.sh to build deeper closures for exactly those targets, then
# re-run this script -- the new files are picked up automatically.
set -euo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" SAILIR_TOPOLOGY=gravity3L

INT=$1; OUT=$2; MODE=${3:-errors}
TAG=$(echo "$INT" | tr ',' '_')          # minus signs PRESERVED (doc §2)
REGEN=results/truth/closures/regen/out
MISSING=${SAILIR_DAGGER_MISSING:-results/truth/closures/regen/missing_targets.txt}
mkdir -p "$(dirname "$MISSING")" logs

CL=""
for D in results/truth/closures/v5_p101_59k/out results/truth/closures/v4_p101_18k/out \
         results/truth/closures/v5_p101_59k/retry results/truth/closures/v4_p101_18k/retry; do
  [ -f "$D/$TAG.json" ] && { CL="$D/$TAG.json"; break; }
done
if [ -z "$CL" ]; then
  # Not fatal any more IF regeneration has already produced closures: the
  # union below can still label states. With neither, there is no expert.
  if [ -d "$REGEN" ] && [ -n "$(ls -A "$REGEN" 2>/dev/null)" ]; then
    echo "NO SHIPPED CLOSURE for $TAG -- proceeding on regen/ only"
  else
    echo "NO CLOSURE for $TAG -- skipping"; exit 0
  fi
fi

# regen dir first so deeper-rung rows win the union
SRC="$REGEN"
[ -n "$CL" ] && SRC="$REGEN:$CL"
export SAILIR_CLOSURE_PROBE="$SRC"
export SAILIR_DAGGER_OUT="$OUT"
export SAILIR_DAGGER_MODE="$MODE"
export SAILIR_DAGGER_MISSING="$MISSING"

python -u reduction/beam_search_v9.py \
    --topology topology_input/gravity3L \
    --model checkpoints/gravity3L_p101_bce/best_model.pt \
    --integral "$INT" --prime 101 --beam-width 20 --max-steps 40 \
    --max-actions 1000 \
    --output /tmp/dagger_$TAG.pkl >> logs/dagger_collect.log 2>&1 || true
