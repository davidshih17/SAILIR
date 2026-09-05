#!/bin/bash
# Train the ftcull model FROM SCRATCH on the gravity3L p=101 corpus.
#
# Implements the recipe in training/TRAIN_FROM_SCRATCH.md:
#   - no --resume        (from scratch, that is the whole point of this branch)
#   - no --backbone_lr   (default 0.0 == one LR for everything; setting it to
#                         1e-5 would train the backbone 10x too slowly, and
#                         there is no pretrained backbone to protect)
#   - --prime 101        MUST match the corpus. Verified: coefficient range
#                        [1,100] with zero coeffs >= 101 across 20 sampled
#                        shards (training/run_verify_corpus.sh).
#   - --lr 1e-4          what the existing from-scratch run used
#   - --lr_schedule plateau
#                        CHANGED from the recipe's implicit cosine. With
#                        CosineAnnealingLR(T_max=200) the LR is still at
#                        97% of peak by epoch 23, so an early plateau gets
#                        no annealing for ~35 more epochs. Observed on this
#                        corpus: val_loss flattened into a 0.323-0.331
#                        oscillation band by E21-E24 while train_loss kept
#                        falling -- LR slightly too high, bouncing in the
#                        basin. ReduceLROnPlateau cuts when val_loss (the
#                        selection objective) actually stalls.
#                        patience=5 > the observed ~3-epoch oscillation
#                        period, so it will not fire on that noise.
#   - --epochs 200       corpus is ~7x the old dots-only one; undertraining is
#                        the risk here, not overfitting
#   - --select_on val_loss + --checkpoint_every 1
#                        see TRAIN_FROM_SCRATCH.md "Checkpoint selection" --
#                        the metric choice is not settled, so every epoch is
#                        kept on disk to allow re-selection later.
#   - --n_val_shards all Use the FULL val split (all 788 shards, ~207K
#                        samples) rather than the 50-shard default. Since
#                        best_model.pt is chosen on val_loss, a noisy val
#                        estimate would make that choice noisy. Costs ~92s per
#                        epoch (~5h over 200 epochs, ~14% of runtime).
#
# NOTE: the recipe in TRAIN_FROM_SCRATCH.md also lists --token_budget 131072.
# That flag is "--data_dir mode only" per its own help text and is a silent
# no-op under --shards_dir, so it is omitted here rather than carried as dead
# weight.
#
# Corpus: data/shard_{0..787}/{train,val}.pt  (~1.51M train / ~207K val samples)
# Topology: gravity3L (n_indices=15, n_denominators=10, n_actions=21)
#
# Re-running this script resumes from checkpoints/gravity3L_p101_scratch/last.pt
# via --auto_resume, so it is safe to relaunch after a crash or a kill.
#
# DO NOT compare this run's val_loss against the historical numbers
# (dots_scratch 1.3205 @ep19, ftcull fine-tune 0.5400 @ep19). Those are on the
# OLD dots-only corpus; this one covers ALL INDICES -- a different problem with
# a different action set, state distribution and loss floor, not a bigger
# version of the same one. val_loss here is only meaningful relative to itself
# (as the curve --select_on val_loss picks a minimum from). The cross-run
# judgement is beam solve behaviour on the 125 test integrals.

set -euo pipefail

cd /home/shih/work/SAILIR_p101

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn

mkdir -p checkpoints/gravity3L_p101_scratch logs

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONUNBUFFERED=1
# Separate PCIe roots on this box (nvidia-smi topo -m shows SYS between all
# pairs), so NCCL P2P is slower than routing through host shared memory.
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN

LOG=logs/train_p101_scratch.log
METRICS=logs/train_p101_scratch_metrics.tsv

echo "Launched: $(date -Iseconds)" | tee -a "$LOG"
echo "Command: $0 $*" | tee -a "$LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"

nohup torchrun --standalone --nnodes=1 --nproc_per_node=3 \
    training/train_classifier.py \
    --topology         topology_input/gravity3L \
    --shards_dir       data \
    --output_dir       checkpoints/gravity3L_p101_scratch \
    --log_file         "$METRICS" \
    --model_variant    nosubs \
    --prime            101 \
    --lr               1e-4 \
    --batch_size       64 \
    --epochs           200 \
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
    --log_every        100 \
    --seed             0 \
    --auto_resume \
    >> "$LOG" 2>&1 &

PID=$!
echo "$PID" > logs/train_p101_scratch.pid
echo "Started PID=$PID. Tail with: tail -f $LOG"
echo "Kill with: kill \$(cat logs/train_p101_scratch.pid)"
