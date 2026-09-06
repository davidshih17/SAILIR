#!/bin/bash
# BEAM CAMPAIGN for the p=101 FROM-SCRATCH model on the 125 test integrals.
#
# Identical to beamexp_worker.sh (the config that produced the first 125/125
# solve) except for the TWO things that must change for this model:
#
#   --model-checkpoint  -> checkpoints/gravity3L_p101_scratch/best_model.pt
#   --prime 1009        -> 101
#
# THE PRIME IS NOT OPTIONAL. This model was trained on a corpus generated with
# the entire elimination at 101, pivots included. Running it at 1009 produces a
# wrong coefficient encoding SILENTLY -- no error, just wrong numbers. Every
# other checkpoint in this repo is 1009, so this is the one worker where the
# default is wrong.
#
# beamexp_worker.sh is NOT edited: it hardcodes the ftcull checkpoint and 1009,
# and several existing .sub files depend on exactly that.
#
#   $1 = target tag   $2 = beam_sort mode   $3 = beam width
set -u
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
export SAILIR_BEAM_SORT=$2
export SAILIR_SCORE=decay
export SAILIR_SCORE_DECAY=0.8
export SAILIR_NM_PENALTY=0.1
export SAILIR_TOP_K=20
export SAILIR_MAX_ACTIONS=1000          # matches the corpus's K=1000 cull
export SAILIR_V9_CULL=1
export SAILIR_V9_UPENUM=1
export SAILIR_RAW_EQ_CACHE_CAP=1000000000
export SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1
export SAILIR_KEEP_CKPT_EVERY=100
ARM=$2$3

# EXPDIR required, no default -- see beamexp_worker.sh for the incident this
# prevents (outputs and logs silently landing in different trees).
if [ -z "${EXPDIR:-}" ]; then
  echo "[worker] FATAL: EXPDIR is unset." >&2
  exit 78
fi
D=$B/results/truth/finetune/${EXPDIR}/$ARM
mkdir -p $D/out $D/ckpt/$1 $D/logs

PYTHONUNBUFFERED=1 exec /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
  $B/reduction/onestep_worker_v9.py \
  --topology $B/topology_input/gravity3L \
  --integral="$(echo "$1" | tr '_' ',')" \
  --output $D/out/$1.pkl \
  --model-checkpoint $B/checkpoints/gravity3L_p101_scratch/best_model.pt \
  --iraws-keep-first 1000000 \
  --checkpoint-path $D/ckpt/$1/ckpt.pkl \
  --checkpoint-interval 100 \
  --beam_width $3 --max_steps 50000 --prime 101 --v7-cpus 1 -v
