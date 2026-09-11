#!/bin/bash
# Reduce the SAME integrals by one IBP worker step; we compare term counts to the
# symmetry rule's tail. --paper-masters-only, same prime/beam as the A/B.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
OUT=$BASE/reduction/ibp_cmp; mkdir -p $OUT
i=0
for INT in "0,-1,2,0,2,-1,0,1,-1,0,0" "0,-1,1,0,2,0,-1,2,0,0,-1" "1,-1,1,0,0,0,0,3,-1,0,0"; do
  i=$((i+1))
  PYTHONUNBUFFERED=1 $PYTHON -u $BASE/reduction/onestep_worker_v7.py \
    --topology $BASE/topology_input/pentagonbox --integral="$INT" \
    --output $OUT/ibp_${i}.pkl --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 --device cpu -v --v7-cpus 1 --paper-masters-only \
    > $OUT/ibp_${i}.log 2>&1
done
