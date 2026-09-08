#!/bin/bash
# DAgger fine-tune: continue the BCE model on base corpus + collected rows.
#
# IDENTICAL to training/run_train_p101_bce.sh except for the four changes
# below, so any difference is attributable to the DAgger data:
#
#   --shards_dir  data_combined -> data_combined2  (adds iteration-2 rows)
#       DAgger prescribes D <- D u D_i. 788 base shards (1.51M samples) plus
#       64 DAgger shards (131,738 rows, 8.7%) linked in as shard_1000+.
#       Training on the DAgger rows ALONE would discard the on-path policy
#       already at 0.9557 val_anyhit, and those rows are `errors`-mode only --
#       adversarial by construction.
#
#   --resume checkpoints/gravity3L_p101_dagger/best_model.pt  (E21, the policy that COLLECTED iteration 2)
#       Epoch 20, the exact checkpoint that generated the rollouts. Starting
#       from anything else would collect from one policy and correct another.
#
#   --restart_lr 1e-5
#       The base run was at 3e-5 by epoch 20 and 9e-6 by epoch 30, i.e. well
#       into plateau decay. 1e-5 is the fine-tune LR used by the ftcull run
#       (TRAIN_FROM_SCRATCH.md); restarting at the original 1e-4 would take a
#       converged model and re-heat it far past where its own schedule had
#       settled.
#
#   --output_dir  a new directory, so best_model.pt for the collection policy
#       stays intact and re-collection stays reproducible.
#
# --select_on val_loss, unchanged and settled.
#
# NOTE ON val_loss COMPARABILITY: --n_val_shards all now covers the COMBINED
# val split, which includes 14,312 DAgger val rows. So val_loss here is NOT
# comparable to the base run's 0.0628 -- different val set, and a harder one
# (off-path states where the model errs 50% of the time vs 4.4% on-path).
# It is only meaningful relative to itself within this run, which is all
# --select_on needs. Cross-run judgement is beam solve behaviour.
set -euo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
mkdir -p checkpoints/gravity3L_p101_dagger2 logs
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export SAILIR_LOSS=bce
LOG=logs/train_p101_dagger2.log
METRICS=logs/train_p101_dagger2_metrics.tsv
echo "Launched: $(date -Iseconds)" | tee -a "$LOG"
echo "Command: $0 $*" | tee -a "$LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  SAILIR_LOSS=$SAILIR_LOSS" | tee -a "$LOG"
echo "shards_dir=data_combined2 (788 base + 64 iter1 + 64 iter2; DAgger share 9.1%)" | tee -a "$LOG"
nohup python -u training/train_classifier.py \
    --topology         topology_input/gravity3L \
    --shards_dir       data_combined2 \
    --resume           checkpoints/gravity3L_p101_dagger/best_model.pt \
    --restart_lr       1e-5 \
    --output_dir       checkpoints/gravity3L_p101_dagger2 \
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
echo "$PID" > logs/train_p101_dagger2.pid
echo "Started PID=$PID. Tail with: tail -f $LOG"
