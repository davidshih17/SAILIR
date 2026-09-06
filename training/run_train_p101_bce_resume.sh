#!/bin/bash
# RESUME the gravity3L p=101 bce run from E30, with two changes:
#
#   --select_on conf_margin   best_model.pt now tracks (best-correct minus
#                             best-incorrect) confidence instead of val_loss.
#                             Why: by E24 val_loss had gone noisy-flat and
#                             anyhit had saturated (0.9597-0.9612), while
#                             conf_margin was still climbing to a run-best
#                             +0.5160 at E30 -- it is the only metric left that
#                             still distinguishes checkpoints. The same score
#                             also drives the plateau scheduler, so the run
#                             anneals on the quantity it is judged by.
#
#   --restart_lr 1e-4         the scheduler had annealed to 9e-6, where progress
#                             continues but slowly. Measured d(conf_max_incorrect)
#                             per epoch: -0.0082 at lr 1e-4, -0.0064 at 3e-5,
#                             -0.0053 at 9e-6 -- monotone in LR with no sign of
#                             instability, so restart at the original 1e-4.
#                             This also resets the plateau scheduler, whose
#                             stored 'best' was accumulated under val_loss and
#                             is meaningless for the new metric.
#
# Every epoch is checkpointed, so an LR-restart shock is recoverable: the E30
# state remains on disk as checkpoint_epoch30.pt.
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
echo "RESUME launched: $(date -Iseconds)  select_on=conf_margin restart_lr=1e-4" | tee -a "$LOG"

nohup python -u training/train_classifier.py \
    --topology         topology_input/gravity3L \
    --shards_dir       data \
    --output_dir       checkpoints/gravity3L_p101_bce \
    --log_file         "$METRICS" \
    --model_variant    nosubs \
    --prime            101 \
    --lr               1e-4 \
    --restart_lr       1e-4 \
    --batch_size       192 \
    --epochs           100 \
    --lr_schedule      plateau \
    --plateau_factor   0.3 \
    --plateau_patience 5 \
    --plateau_threshold 1e-3 \
    --plateau_min_lr   1e-6 \
    --select_on        conf_margin \
    --checkpoint_every 1 \
    --num_workers      4 \
    --buffer_shards    6 \
    --n_val_shards     all \
    --log_every        200 \
    --seed             0 \
    --auto_resume \
    >> "$LOG" 2>&1 &
PID=$!; echo "$PID" > logs/train_p101_bce.pid
echo "Started PID=$PID"
