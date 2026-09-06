#!/bin/bash
# Combine run 1 + run 2 into ONE recdir and submit the sharded pack.
#
# pack_shard.py takes a single --recdir, and the README (section 6) says a
# shard-packed corpus is not bit-comparable with a piecewise one -- "do not mix;
# repack whole". So both runs are symlinked into one directory and packed in a
# single pass. Symlinks, not copies: the corpus is ~30 GB.
#
# Safe because the two runs are DISJOINT by construction (run 2's sampling used
# --exclude on run 1's targets) and VERIFIED: 0 overlapping filenames. crc32
# shard assignment works on the basename, so a combined pack splits cleanly.
#
# RUN THIS ONLY AFTER the walk queue has drained. Packing while jobs are still
# appending captures partial trajectories.
set -u
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
R1=$B/results/truth/p101_cull_corpus/train
R2=$B/results/truth/p101_cull_corpus_v2/train
ALL=$B/results/truth/p101_corpus_all
SHARDS=$B/results/truth/p101_shards

q=$(timeout 60 condor_q "$USER" -totals 2>/dev/null | grep -oE 'query: [0-9]+' | grep -oE '[0-9]+')
if [ -z "$q" ]; then echo "ABORT: schedd unreachable"; exit 1; fi
if [ "$q" != "0" ]; then echo "ABORT: $q jobs still queued -- wait for the walks to finish"; exit 1; fi

mkdir -p "$ALL" "$SHARDS/logs"
# Links MUST be named rec_<tag>.jsonl: pack_shard.py globs for 'rec_*.jsonl'
# (line ~82). The truthcull worker writes <tag>.jsonl with NO prefix -- the
# rec_ prefix came from the older truth_record_variant.py worker. Without this
# the pack matches ZERO files and every shard job exits with
# "no non-empty recordings under ..." (measured: all 788 jobs, 0 shards).
# Renaming here rather than changing pack_shard.py, which older corpora share.
#
# find, not a glob: these dirs hold 12k-40k files, past ARG_MAX (PITFALL 12).
find "$R1" "$R2" -name '*.jsonl' -printf '%p\0' \
  | xargs -0 -I{} sh -c 'ln -sf "$1" "$2/rec_$(basename "$1")"' _ {} "$ALL"

n=$(find "$ALL" -name 'rec_*.jsonl' | wc -l)
n1=$(find "$R1" -name '*.jsonl' | wc -l)
n2=$(find "$R2" -name '*.jsonl' | wc -l)
echo "run1=$n1  run2=$n2  combined=$n"
if [ "$n" -ne $((n1 + n2)) ]; then
    echo "ABORT: combined $n != $n1 + $n2 -- filename collision between runs"
    exit 1
fi

# Shard count sized by ROWS, matching the reference's ~2,772 rows/shard
# (ftcull1848_corrupt: 210,656 rows over 76 shards). Our files are far smaller
# per target, so a file-count split would make shards too light.
# `xargs cat | wc -l` takes 25+ MINUTES on 52k files -- the pipe is the
# bottleneck. `xargs wc -l` lets wc read each file directly: 89 SECONDS.
rows=$(find "$ALL" -name 'rec_*.jsonl' -print0 | xargs -0 wc -l 2>/dev/null | awk '$2=="total"{s+=$1} END{print s}')
NS=$(( (rows + 2771) / 2772 ))
[ "$NS" -lt 1 ] && NS=1
echo "rows=$rows -> nshards=$NS"

cat > "$SHARDS/pack.sub" <<EOF
universe   = vanilla
executable = $B/data-gen/run_pack_shard.sh
arguments  = "\$(Process)"

# 4 GB/core (README 1.3b): request_cpus = ceil(request_memory / 4GB).
# 8 GB -> 2 cpus. Measured 2.99 GB peak on a 48-file shard of rand_corpus.
request_cpus   = 2
request_memory = 8GB
request_disk   = 8GB

# --keep-subs deliberately NOT set: the nosubs model drops every sub_* tensor on
# the first line of forward(), and they are 44.6% of packed bytes.
environment = "RECDIR=$ALL OUTDIR=$SHARDS NSHARDS=$NS SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1"

log    = $SHARDS/logs/pack_\$(ClusterId)_\$(Process).log
output = $SHARDS/logs/pack_\$(ClusterId)_\$(Process).out
error  = $SHARDS/logs/pack_\$(ClusterId)_\$(Process).err

queue $NS
EOF
echo "wrote $SHARDS/pack.sub"
echo "submit with: condor_submit $SHARDS/pack.sub"
