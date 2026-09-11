#!/bin/bash
# Run the v7 one-step worker on one integral with SAILIR_SYMMETRY off and on,
# in parallel. Args: <integral> <tag> [cpus]. Output + logs in reduction/sym_test/.
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
INT="$1"; TAG="$2"; CPUS="${3:-4}"
CK=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPO=$BASE/topology_input/pentagonbox
OUT=$BASE/reduction/sym_test
mkdir -p "$OUT"
cd "$BASE/reduction" || exit 1
common="--topology $TOPO --integral=$INT --model-checkpoint $CK --beam_width 20 --paper-masters-only --v7-cpus $CPUS --verbose"

SAILIR_SYMMETRY=0 PYTHONUNBUFFERED=1 $PY onestep_worker_v7.py $common \
    --output "$OUT/${TAG}_off.pkl" > "$OUT/${TAG}_off.log" 2>&1 &
P0=$!
SAILIR_SYMMETRY=1 PYTHONUNBUFFERED=1 $PY onestep_worker_v7.py $common \
    --output "$OUT/${TAG}_on.pkl" > "$OUT/${TAG}_on.log" 2>&1 &
P1=$!
echo "OFF pid $P0  ON pid $P1  tag=$TAG int=$INT cpus=$CPUS"
wait $P0 $P1
echo "=== both done ==="
