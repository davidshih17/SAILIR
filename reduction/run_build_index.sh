#!/bin/bash
# Build the full recipe index over ALL worker results, sharded.
# 128 cores / 319 GB available; 8 shards is a small slice of the login node
# and the job is pure analysis of finished results (no Condor contention with
# the live campaign, which has 8,650 jobs already idle).
set -u
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
eval "$(/het/p4/dshih/conda_envs/pyg4/bin/python scripts/greedy_env.py --export 2>/dev/null)"
export SAILIR_SYM_STORE=results/gravity3L_transforms_v2_p101.pkl
export SAILIR_RAW_EQ_CACHE_CAP=100000
export SAILIR_FAST_COEFF=1
export PYTHONUNBUFFERED=1
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
OUT=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/recipe_shards
mkdir -p "$OUT"
N=8
for i in $(seq 0 $((N-1))); do
  $PY reduction/build_recipe_index.py "$OUT/shard_$i.bin" \
      --shard "$i" --nshards "$N" > "$OUT/shard_$i.log" 2>&1 &
  echo "  launched shard $i pid $!"
done
wait
echo "ALL SHARDS DONE $(date +%H:%M:%S)"
$PY reduction/build_recipe_index.py \
    /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/recipe_index_full.npz \
    --merge "$OUT"/shard_*.bin
echo "MERGE DONE $(date +%H:%M:%S)"
