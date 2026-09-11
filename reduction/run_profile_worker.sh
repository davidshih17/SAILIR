#!/bin/bash
# Profile one real worker search with the live campaign's flags.
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
eval "$(/het/p4/dshih/conda_envs/pyg4/bin/python scripts/greedy_env.py --export 2>/dev/null)"
export SAILIR_SYM_STORE=results/gravity3L_transforms_v2_p101.pkl
export PYTHONUNBUFFERED=1
export PROF_INTEGRAL='0,-10,-6,1,1,1,1,1,0,1,-1,-1,0,0,0'
export PROF_STEPS=120
exec /het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python \
  /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/reduction/profile_worker.py
