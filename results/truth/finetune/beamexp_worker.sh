#!/bin/bash
# BEAM-CONFIGURATION EXPERIMENT on the hard tail, ftcorrupt model.
#
# WHY. ftcorrupt solves 97/100 and takes 19% fewer steps in aggregate, but is
# WORSE on the hard tail: 1_2_1_1_1_1_1_1_0_1 passed ftcull4's 766 steps and is
# still running, and the pathological 1_0_1_0_0_0_5_1_1_1 is far past corr13's
# 215. The production beam is DUAL: two sub-beams of 40, one ranked by weight
# (lane 0) and one by model probability (lane 1), each descending only from
# itself so neither can starve the other. Of 97 solves, 52 finished on lane 0
# and 45 on lane 1 -- both lanes earn their keep on easy targets, but that says
# nothing about which one is carrying the hard ones.
#
# ARMS (each 40 states unless noted):
#   weight  lane 0 alone   -- pure weight ordering, no probability lane
#   prob    lane 1 alone   -- pure model-probability ordering
#   dual80  both lanes, 80 states -- twice the width, same split
#   dual40  both lanes, 40 states -- THE CONTROL
#
# dual40 is included deliberately even though the live campaign already runs
# that config: those jobs started at different times on different machines and
# are hours ahead, so they are not a clean baseline for arms launched now. An
# experiment whose control is a differently-conditioned run measures the
# conditions, not the arms.
#
# Checkpointing is on (~6 KB per step of depth) so any arm that goes deep
# leaves a ladder to profile instead of having to be re-run.
#
#   $1 = target tag   $2 = beam_sort mode   $3 = beam width
set -u
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
export SAILIR_BEAM_SORT=$2
export SAILIR_SCORE=decay
export SAILIR_SCORE_DECAY=0.8
export SAILIR_NM_PENALTY=0.1
export SAILIR_TOP_K=20
export SAILIR_MAX_ACTIONS=1000
export SAILIR_V9_CULL=1
export SAILIR_V9_UPENUM=1
export SAILIR_RAW_EQ_CACHE_CAP=1000000000
export SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1
export SAILIR_KEEP_CKPT_EVERY=100
ARM=$2$3
# EXPDIR is REQUIRED, no default. It used to default to `beamexp`, which
# silently decoupled a run's OUTPUTS from its LOGS: the .sub file hardcodes the
# log path, the worker derived the output path from EXPDIR, and nothing checked
# they agreed. beamexp25.sub shipped without the environment line, so on
# 2026-08-31 its results landed in beamexp/ while its logs landed in beamexp25/
# -- 12 solved targets looked "missing" from beamexp25/prob40/out for two days,
# and beamexp/prob40 became a silent mixture of two campaigns. A redo submit
# added the line mid-flight, which is why that run's outputs are split across
# two trees to this day. Failing loudly here costs one line; the silent version
# cost a wrong comparison and an afternoon.
if [ -z "${EXPDIR:-}" ]; then
  echo "[worker] FATAL: EXPDIR is unset. Add to the .sub file:" >&2
  echo "           environment = \"EXPDIR=<dir matching the log path>\"" >&2
  exit 78
fi
D=$B/results/truth/finetune/${EXPDIR}/$ARM
mkdir -p $D/out $D/ckpt/$1 $D/logs
PYTHONUNBUFFERED=1 exec /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
  $B/reduction/onestep_worker_v9.py \
  --topology $B/topology_input/gravity3L \
  --integral="$(echo "$1" | tr '_' ',')" \
  --output $D/out/$1.pkl \
  --model-checkpoint $B/checkpoints/gravity3L_dots_ftcull_corrupt/best_model.pt \
  --iraws-keep-first 1000000 \
  --checkpoint-path $D/ckpt/$1/ckpt.pkl \
  --checkpoint-interval 100 \
  --beam_width $3 --max_steps 50000 --prime 1009 --v7-cpus 1 -v
