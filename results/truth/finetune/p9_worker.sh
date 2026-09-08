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
# SAILIR_BEAM_SORT removed: no beam_sort parameter remains
# Overridable. Default 'decay' keeps every existing .sub byte-identical.
# SAILIR_SCORE=local scores a state by the log-probability of the LAST
# action only. MEASURED reason to want it: with tabu off, at depth 9 the
# whole beam ties at max_w12 [10,1], so the score tiebreak decides alone --
# and the truth path loses on a decayed debt from its depth-2 step
# (p=0.0025, ~6 nats, still ~1.3 after decay) despite its CURRENT move
# being rank 2 at p=0.113. Local scoring drops the debt.
# SAILIR_SCORE removed: score mode folded to local
# Overridable. Default 0.1 keeps every existing .sub byte-identical.
# THIS LINE WAS UNCONDITIONAL until 2026-09-07 and silently overrode
# SAILIR_NM_PENALTY set in the submit file -- so the 125/125 and 300/300
# greedy runs, which I described as 'plain argmax', actually selected on
# (action log-prob - 0.1 * n_non_masters). Their banners say NM_PEN=0.1.
export SAILIR_NM_PENALTY=${SAILIR_NM_PENALTY:-0.1}
# SAILIR_TOP_K removed: top_k=20 is now a real parameter default
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

# THREE CAPS so a runaway greedy walk cannot spiral. Greedy has no beam width
# bounding it, so a walk that never drains the bucket just grows the
# substitution store until the node suffers.
#   SAILIR_BEAM_WALL  wall clock, default 1800s. The known solve took 10.6s.
#   SAILIR_MAX_STEPS  step cap, default 50000 (= the previous hardcoded value,
#                     so existing .sub files are unchanged).
# Memory has NO in-process cap in onestep_worker_p9.py (unlike the walk
# worker), so it is bounded by the Condor reservation + periodic_hold instead.
#
# `timeout` execs its command directly, so an inline VAR=VAL in front of the
# interpreter would be treated as the command name and fail with "No such file
# or directory". Export first. (This exact bug cost a run earlier.)
export PYTHONUNBUFFERED=1
exec timeout "${SAILIR_BEAM_WALL:-1800}" \
  /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
  $B/reduction/onestep_worker_p9.py \
  --topology $B/topology_input/gravity3L \
  --integral="$(echo "$1" | tr '_' ',')" \
  --output $D/out/$1.pkl \
  --model-checkpoint $B/checkpoints/gravity3L_p101_scratch/best_model.pt \
  --checkpoint-path $D/ckpt/$1/ckpt.pkl \
  --checkpoint-interval 100 \
  --beam_width $3 --max_steps ${SAILIR_MAX_STEPS:-50000} --prime 101 --v7-cpus 1 -v \
  ${SAILIR_EXTRA_FLAGS:-}
# SAILIR_EXTRA_FLAGS is EMPTY by default, so every existing .sub that uses
# this worker runs byte-identically. It exists so a flag that has no env
# equivalent -- notably --no-tabu -- can be set per submit file without
# forking the worker.
