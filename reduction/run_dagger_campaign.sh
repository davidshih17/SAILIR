#!/bin/bash
# DAgger campaign: roll the model out over many targets, label the states IT
# visits with the corpus's own expert (unused closure rows), and emit jsonl
# that data-gen/preprocess_to_tensors.py packs unchanged.
#
#   $1  file of integrals, one per line (comma-separated)
#   $2  output jsonl
#   $3  mode: errors (default) | all
#
#
# --max-actions 1000 MATCHES the corpus cull (K=1000, verified: max 1000
# actions/state, 33% at the cap). The beam default is 900, which would collect
# rows in a slightly different action space than the model was trained in.
# Skips targets with no closure (the held-out hard set) rather than failing.
set -euo pipefail
cd /home/shih/work/SAILIR_p101
TARGETS=$1; OUT=$2; MODE=${3:-errors}
: > "$OUT"; : > logs/dagger_collect.log
n=0; skipped=0
while read -r INT; do
  [ -z "$INT" ] && continue
  before=$(wc -l < "$OUT")
  ./reduction/run_dagger_collect.sh "$INT" "$OUT" "$MODE" >/dev/null 2>&1 || true
  after=$(wc -l < "$OUT")
  n=$((n+1))
  echo "  [$n] $INT -> +$((after-before)) rows (total $after)"
done < "$TARGETS"
echo
echo "targets processed: $n"
echo "rows emitted     : $(wc -l < "$OUT")"
