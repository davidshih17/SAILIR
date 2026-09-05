#!/bin/bash
# Smoke test for the gravity3L p=101 from-scratch run.
#
# Per training/README.md §9: 64 train shards / 16 val shards / 5 epochs.
# Expect val top-1 to climb past ~40% by epoch 5.
#
# Purpose: prove the whole pipeline end-to-end (manifest build, DDP init,
# sharded streaming, train->val boundary, checkpoint save, epoch transitions)
# BEFORE committing to the ~2-3 day 200-epoch run.
#
# Isolated output: checkpoints/smoke_p101/ + logs/smoke_test_p101.log
# Detached via nohup so it survives session disconnect.

set -euo pipefail

cd /home/shih/work/SAILIR_p101

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn

mkdir -p checkpoints logs

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONUNBUFFERED=1
# Separate PCIe roots on this box (nvidia-smi topo -m shows SYS between all
# pairs), so NCCL P2P is slower than routing through host shared memory.
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN

LOG=logs/smoke_test_p101.log
: > "$LOG"

echo "Smoke test launched: $(date -Iseconds)" | tee -a "$LOG"
echo "Command: $0 $*" | tee -a "$LOG"

nohup torchrun --standalone --nnodes=1 --nproc_per_node=3 \
    training/train_classifier.py \
    --topology         topology_input/gravity3L \
    --shards_dir       data \
    --output_dir       checkpoints/smoke_p101 \
    --model_variant    nosubs \
    --prime            101 \
    --lr               1e-4 \
    --batch_size       64 \
    --max_train_shards 64 \
    --n_val_shards     16 \
    --epochs           5 \
    --select_on        val_loss \
    --checkpoint_every 1 \
    --num_workers      4 \
    --buffer_shards    6 \
    --log_every        50 \
    --seed             0 \
    >> "$LOG" 2>&1 &

PID=$!
echo "$PID" > logs/smoke_test_p101.pid
echo "Started PID=$PID. Tail with: tail -f $LOG" | tee -a "$LOG"
echo "Kill with: kill \$(cat logs/smoke_test_p101.pid)" | tee -a "$LOG"
