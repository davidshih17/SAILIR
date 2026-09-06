#!/bin/bash
# BCE (multi-label) training of the gravity3L p=101 model -- SINGLE GPU.
#
# A/B partner to training/run_train_p101_scratch.sh (the CE run, best_model.pt
# at E34). Everything is held equal except the objective:
#
#   SAILIR_LOSS=bce   independent per-action sigmoid; every VALID action -> 1,
#                     every other -> 0. Doubly normalised so the objective is
#                     invariant to how many actions a state offers and to its
#                     positive/negative ratio (per-state pos_weight, NOT a
#                     global one -- the measured per-state ratio spans ~11:1 to
#                     ~999:1 around a 109:1 mean). Verified by
#                     training/test_bce_loss.py.
#                     Sets score_activation=sigmoid automatically, so the beam
#                     does not re-impose softmax's action-count coupling.
#
# SINGLE GPU, and why: GPUs 1 and 2 are occupied by another user (2x ~34 GB at
# 100% SM utilisation). A 3-rank DDP run OOM'd against that. GPU 0 is entirely
# free, so this trades ~4x wall-clock for a run that cannot be killed by a
# neighbour's allocation growing.
#
# --batch_size 192 is NOT the CE run's 64: that run was 64/rank x 3 ranks, so
# its EFFECTIVE batch was 192. Matching the effective batch keeps the optimiser
# dynamics comparable between the two arms.
#
# METRICS -- for a bce model, top1/top5/top20 are MEANINGLESS: they score
# agreement with the single arbitrarily-recorded label, which this objective
# deliberately does not target. Read instead:
#     conf_best_correct   mean over states of max sigma over CORRECT actions
#                         -- "is it confident on AT LEAST ONE correct action",
#                         which is all the beam needs. THE headline number.
#     conf_any50/any90    the same as a RATE: fraction of states where some
#                         correct action clears 0.5 / 0.9.
#     conf_max_incorrect  the failure mode -- confidently wrong. Should fall.
#     conf_margin         best correct minus worst incorrect.
#     anyhit / anyhit20   rank-based but scored against the FULL valid set,
#                         so still meaningful (unlike top1).
# val_loss is the BCE objective and is NOT comparable to the CE run's val_loss.
# Cross-arm comparison must use anyhit / conf_* / beam solve rate.

set -euo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
mkdir -p checkpoints/gravity3L_p101_bce logs

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export SAILIR_LOSS=bce

LOG=logs/train_p101_bce.log
METRICS=logs/train_p101_bce_metrics.tsv

echo "Launched: $(date -Iseconds)" | tee -a "$LOG"
echo "Command: $0 $*" | tee -a "$LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  SAILIR_LOSS=$SAILIR_LOSS" | tee -a "$LOG"

nohup python -u training/train_classifier.py \
    --topology         topology_input/gravity3L \
    --shards_dir       data \
    --output_dir       checkpoints/gravity3L_p101_bce \
    --log_file         "$METRICS" \
    --model_variant    nosubs \
    --prime            101 \
    --lr               1e-4 \
    --batch_size       192 \
    --epochs           100 \
    --lr_schedule      plateau \
    --plateau_factor   0.3 \
    --plateau_patience 5 \
    --plateau_threshold 1e-3 \
    --plateau_min_lr   1e-6 \
    --select_on        val_loss \
    --checkpoint_every 1 \
    --num_workers      4 \
    --buffer_shards    6 \
    --n_val_shards     all \
    --log_every        200 \
    --seed             0 \
    --auto_resume \
    >> "$LOG" 2>&1 &

PID=$!
echo "$PID" > logs/train_p101_bce.pid
echo "Started PID=$PID. Tail with: tail -f $LOG"
echo "Kill with: kill \$(cat logs/train_p101_bce.pid)"
