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
PYBIN=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
# ── CERTIFIED CONFIGURATION: read from the record, never re-typed here ──
# reduction/greedy_certified.json is the single source of truth, shared with
# greedy_reduce.py (which aborts at import if the env is wrong) and
# hierarchical_reduction.py (which pins the same env into every Condor submit).
# This wrapper used to carry its own hand-maintained copy of the exports, the
# prime, the cpu count and the checkpoint -- a third copy that could drift from
# the other two with nothing comparing them.
#
# --unset matters as much as --export: it clears every variable the certified
# run had UNSET, so one inherited from the submitting shell cannot leak in.
# SAILIR_SYM_FIRST=1 in particular would make the worker return a symmetry rule
# with steps=0 and never run the search, while every exported value stayed right.
eval "$($PYBIN $B/scripts/greedy_env.py --unset)"
eval "$($PYBIN $B/scripts/greedy_env.py --export)"
eval "$($PYBIN $B/scripts/greedy_env.py --cli)"

# Campaign knobs, deliberately NOT in the record: they bound how long a run may
# take, never which path it walks.
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
# Memory has NO in-process cap in greedy_worker.py (unlike the walk
# worker), so it is bounded by the Condor reservation + periodic_hold instead.
#
# `timeout` execs its command directly, so an inline VAR=VAL in front of the
# interpreter would be treated as the command name and fail with "No such file
# or directory". Export first. (This exact bug cost a run earlier.)
export PYTHONUNBUFFERED=1
exec timeout "${SAILIR_BEAM_WALL:-1800}" \
  /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
  $B/reduction/greedy_worker.py \
  --topology $B/topology_input/gravity3L \
  --integral="$(echo "$1" | tr '_' ',')" \
  --output $D/out/$1.pkl \
  --model-checkpoint $B/$GREEDY_CKPT \
  --checkpoint-path $D/ckpt/$1/ckpt.pkl \
  --checkpoint-interval 100 \
  --beam_width $3 --max_steps ${SAILIR_MAX_STEPS:-50000} --prime $GREEDY_PRIME --v7-cpus $GREEDY_V7_CPUS -v \
  ${SAILIR_EXTRA_FLAGS:-}
# SAILIR_EXTRA_FLAGS is EMPTY by default, so every existing .sub that uses
# this worker runs byte-identically. It exists so a flag that has no env
# equivalent -- notably --no-tabu -- can be set per submit file without
# forking the worker.
