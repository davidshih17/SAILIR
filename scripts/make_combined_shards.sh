#!/bin/bash
# Build the AGGREGATED training set for DAgger fine-tuning.
#
# DAgger prescribes D <- D u D_i: train on the base corpus PLUS the collected
# rows, never the new rows alone. Training on the DAgger shards by themselves
# would discard the on-path policy that is already at 0.9557 val_anyhit, and
# those rows are `errors`-mode only -- adversarial by construction.
#
# The loader globs <shards_dir>/shard_*/{train,val}.pt, so base shard_0..787
# and dagger shard_0..63 WOULD COLLIDE. Dagger shards are linked in as
# shard_1000+ to keep both sets addressable.
#
# Symlinks, not copies: the base corpus is large and this is reversible.
set -uo pipefail
cd /home/shih/work/SAILIR_p101

OUT=${1:-data_combined}
rm -rf "$OUT"; mkdir -p "$OUT"

n_base=0
for D in data/shard_*; do
  [ -d "$D" ] || continue
  ln -s "$(readlink -f "$D")" "$OUT/$(basename "$D")"
  n_base=$((n_base+1))
done

n_dag=0
for D in data_dagger/shard_*; do
  [ -d "$D" ] || continue
  N=$(basename "$D" | sed 's/shard_//')
  ln -s "$(readlink -f "$D")" "$OUT/shard_$((1000+N))"
  n_dag=$((n_dag+1))
done

echo "combined dir : $OUT"
echo "  base shards   : $n_base"
echo "  dagger shards : $n_dag  (linked as shard_1000+)"
echo "  total links   : $(ls "$OUT" | wc -l)"
# no stale manifest: counts would be wrong for the merged set
rm -f "$OUT/manifest.json"
