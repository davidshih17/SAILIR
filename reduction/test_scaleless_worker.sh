#!/bin/bash
# Does an IBP worker reduce a scaleless (trivial-sector) integral to 0 on its own?
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
OUT=$BASE/reduction/scaleless_test; mkdir -p $OUT
for INT in "0,0,0,0,1,1,0,1,0,0,0" "0,0,1,0,1,1,0,0,0,0,0" "1,0,0,0,1,1,0,0,0,0,0"; do
  for PM in "--paper-masters-only" "--no-paper-masters-only"; do
    tag=$(echo "${INT}_${PM}" | tr ',=- ' '____')
    PYTHONUNBUFFERED=1 $PYTHON -u $BASE/reduction/onestep_worker_v7.py \
      --topology $BASE/topology_input/pentagonbox --integral="$INT" \
      --output $OUT/w_${tag}.pkl --model-checkpoint $MODEL \
      --beam_width 40 --max_steps 1000000 --prime 1009 --device cpu -v --v7-cpus 1 $PM \
      > $OUT/w_${tag}.log 2>&1
    echo "=== I[$INT]  $PM ==="
    grep -iE "SUCCESS|FAIL|master|final|reduce|result|zero|empty" $OUT/w_${tag}.log | tail -4
    echo ""
  done
done
