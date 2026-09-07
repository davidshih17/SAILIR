#!/bin/bash
# Split round-3 DAgger rows into per-trajectory rec_*.jsonl, then pack to shards.
#
# Two properties the packer depends on (TRAIN_FROM_SCRATCH.md):
#   - split is BY FILE, so a trajectory never straddles train/val
#   - shard assignment is crc32 of the FILENAME, not python hash()
# split_dagger_rows.py produces exactly that layout and fixes start_target.
#
# --keep-subs stays OFF: the nosubs model deletes every sub_* tensor on the
# first line of forward(), yet they were 44.6% of packed bytes.
set -uo pipefail
cd /home/shih/work/SAILIR_p101
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g4pinn
export PYTHONUNBUFFERED=1 SAILIR_TOPOLOGY=gravity3L

R=${1:-$(ls -td results/dagger/round_* | head -1)}
OUT=${2:-data_dagger}
NSHARDS=${3:-64}
JOBS=${4:-8}

echo "=== pack DAgger rows  $(date -Iseconds) ==="
echo "Command: $0 $*"
echo "round=$R  outdir=$OUT  nshards=$NSHARDS  jobs=$JOBS"

echo "--- split into per-trajectory rec_*.jsonl ---"
python -u scripts/split_dagger_rows.py --run "$R" --outdir "$R/rec"
echo "rec files: $(ls "$R/rec" | wc -l)"

mkdir -p "$OUT"
echo "--- pack $NSHARDS shards ($JOBS at a time) ---"
seq 0 $((NSHARDS-1)) | xargs -P "$JOBS" -I{} bash -c '
  python -u data-gen/pack_shard.py \
      --topology topology_input/gravity3L \
      --recdir "'"$R"'/rec" --outdir "'"$OUT"'" \
      --shard {} --nshards '"$NSHARDS"' --val-every 10 \
      > logs/pack_dagger_shard_{}.log 2>&1 \
    && echo "shard {} OK" || echo "shard {} FAILED"
'
echo "--- result ---"
ls "$OUT" | head -3
echo "shards packed: $(ls -d $OUT/shard_* 2>/dev/null | wc -l)"
du -sh "$OUT"
