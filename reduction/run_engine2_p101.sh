#!/bin/bash
# Rebuild the GENERAL-engine transform store (the one gravity3L ACTUALLY uses)
# at the model's prime.
#
# WHICH STORE. symmetry_route uses topo_config.canonicalize_module(), which for
# gravity3L is canonicalize2 -- NOT canonicalize_GR -- and canonicalize2 reads
# topo_config.STORE_PKL = results/gravity3L_transforms_v2.pkl, deriving its
# prime from that store's prod_point. results/gr_transforms.pkl is a DIFFERENT
# store (the GR-specific engine) used as the gate reference. Rebuilding
# gr_transforms at p=101 therefore did nothing for routing; this is the file
# that matters.
#
# CONTROL FIRST. Build at 1009 to a scratch path and compare against the
# existing store. If the invocation is faithful the two agree, which is what
# makes the 101 build trustworthy -- the same discipline that validated the GR
# rebuild (control was byte-identical there).
set -u
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
SYM=$B/topology_input/gravity3L/kira_validate/sectormappings/GR
cd $B
mkdir -p reduction/logs

echo "############ CONTROL build at 1009 ############"
PYTHONUNBUFFERED=1 $PY reduction/symmetry_engine2.py \
    topology_input/gravity3L "$SYM" \
    results/gravity3L_transforms_v2_ctrl1009.pkl 1009
echo "control rc=$?"

echo
echo "############ compare control vs the in-use store ############"
$PY - <<'PY'
import pickle
a=pickle.load(open('results/gravity3L_transforms_v2.pkl','rb'))
b=pickle.load(open('results/gravity3L_transforms_v2_ctrl1009.pkl','rb'))
print('  identical:', a==b)
print('  prod_point:', a.get('prod_point'), '->', b.get('prod_point'))
print('  sectors:', len(a.get('by_sector',{})), '->', len(b.get('by_sector',{})))
PY

echo
echo "############ BUILD at 101 ############"
PYTHONUNBUFFERED=1 $PY reduction/symmetry_engine2.py \
    topology_input/gravity3L "$SYM" \
    results/gravity3L_transforms_v2_p101.pkl 101
echo "p101 rc=$?"

echo
echo "############ p101 store summary ############"
$PY - <<'PY'
import pickle, os
f='results/gravity3L_transforms_v2_p101.pkl'
if not os.path.exists(f):
    print('  NOT BUILT'); raise SystemExit
d=pickle.load(open(f,'rb'))
print('  prod_point:', d.get('prod_point'))
print('  sectors   :', len(d.get('by_sector',{})))
print('  transforms:', sum(len(v) for v in d.get('by_sector',{}).values()))
PY
echo "ENGINE2 P101 DONE"
