#!/bin/bash
# DAGGER FEASIBILITY PROBE.
#
# Runs the model-guided v9 beam on a target whose truth closure we have, and at
# every visited state counts how many of its LEGAL actions are UNUSED CLOSURE
# ROWS -- the exact condition truthminnew needs to produce a label.
#
#   expert_can_label = tasks where at least one unused closure row is legal
#   exhausted        = tasks where truthminnew has NOTHING -> no DAgger label
#
# If frac_labelable stays high as the beam deviates, DAgger is cheap and covered.
# If it collapses, most off-path states are unlabelable and the only usable
# signal is the abstention one (closure exhaustion == evidenced dead end).
set -euo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
mkdir -p logs results
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" SAILIR_TOPOLOGY=gravity3L
export SAILIR_CONF_PROBE=1

INT=$1; TAG=$(echo "$INT" | tr ',' '_')     # minus signs PRESERVED (doc §2)
# search both campaigns, out/ then retry/
CL=""
for D in results/truth/closures/v5_p101_59k/out results/truth/closures/v4_p101_18k/out \
         results/truth/closures/v5_p101_59k/retry results/truth/closures/v4_p101_18k/retry; do
  [ -f "$D/$TAG.json" ] && { CL="$D/$TAG.json"; break; }
done
[ -n "$CL" ] || { echo "NO CLOSURE for tag $TAG"; exit 1; }
export SAILIR_CLOSURE_PROBE="$CL"
echo "closure: $CL"

python -u reduction/beam_search_v9.py \
    --topology topology_input/gravity3L \
    --model checkpoints/gravity3L_p101_bce/best_model.pt \
    --integral "$INT" --prime 101 --beam-width 20 --max-steps 40 \
    --output /tmp/dagger_probe.pkl > logs/dagger_feasibility.log 2>&1 || true
grep -E "CLOSUREPROBE|SUCCESS|DONE" logs/dagger_feasibility.log | head -30
