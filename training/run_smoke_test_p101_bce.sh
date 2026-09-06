#!/bin/bash
# BCE smoke test -- A/B against the CE smoke test run earlier.
#
# IDENTICAL config to training/run_smoke_test_p101.sh (50 train shards, 9 val
# shards, 5 epochs, lr 1e-4, bs 64, cosine T_max=5) so the ONLY variable is
# SAILIR_LOSS=bce. CE baseline from that run, for comparison:
#     E1 top1=0.4707 top5=0.8519 top20=0.9599
#     E3 top1=0.6620 top5=0.9460 top20=0.9877
#     E5 top1=0.7191 top5=0.9568 top20=0.9877
#
# NOTE: val_loss is NOT comparable between the two -- they are different
# objectives. Compare top1 / top5 / top20 / anyhit.
set -euo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
mkdir -p checkpoints logs

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export SAILIR_LOSS=bce          # <-- the only change vs the CE smoke test

LOG=logs/smoke_test_p101_bce.log
: > "$LOG"
echo "BCE smoke test launched: $(date -Iseconds)" | tee -a "$LOG"

nohup torchrun --standalone --nnodes=1 --nproc_per_node=3 \
    training/train_classifier.py \
    --topology         topology_input/gravity3L \
    --shards_dir       data \
    --output_dir       checkpoints/smoke_p101_bce \
    --log_file         logs/smoke_test_p101_bce_metrics.tsv \
    --model_variant    nosubs \
    --prime            101 \
    --lr               1e-4 \
    --batch_size       64 \
    --max_train_shards 50 \
    --n_val_shards     9 \
    --epochs           3 \
    --select_on        val_loss \
    --checkpoint_every 1 \
    --num_workers      4 \
    --buffer_shards    6 \
    --log_every        50 \
    --seed             0 \
    >> "$LOG" 2>&1 &

PID=$!
echo "$PID" > logs/smoke_test_p101_bce.pid
echo "Started PID=$PID. Tail: tail -f $LOG" | tee -a "$LOG"
