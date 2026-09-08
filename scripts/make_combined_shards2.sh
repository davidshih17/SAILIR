#!/bin/bash
# Aggregate for DAgger iteration 2:  D <- base u D_1 u D_2
#
# The loader globs <shards_dir>/shard_*/{train,val}.pt and each source numbers
# its shards from 0, so they are offset into disjoint ranges:
#   base        shard_0..787
#   iteration 1 shard_1000+
#   iteration 2 shard_2000+
# Symlinks, not copies -- reversible and the base corpus is large.
set -uo pipefail
cd /home/shih/work/SAILIR_p101
OUT=${1:-data_combined2}
rm -rf "$OUT"; mkdir -p "$OUT"
n=0
for D in data/shard_*; do [ -d "$D" ] || continue
  ln -s "$(readlink -f "$D")" "$OUT/$(basename "$D")"; n=$((n+1)); done
echo "  base        : $n"
for SRC in "data_dagger:1000" "data_dagger2:2000"; do
  DIR=${SRC%%:*}; OFF=${SRC##*:}; m=0
  for D in $DIR/shard_*; do [ -d "$D" ] || continue
    N=$(basename "$D" | sed 's/shard_//')
    ln -s "$(readlink -f "$D")" "$OUT/shard_$((OFF+N))"; m=$((m+1)); done
  echo "  $DIR : $m  (as shard_${OFF}+)"
done
rm -f "$OUT/manifest.json"
echo "  total links : $(ls "$OUT" | wc -l)"
echo "  duplicates  : $(ls "$OUT" | sort | uniq -d | wc -l)"
