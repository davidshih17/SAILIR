#!/bin/bash
# ============================================================================
# (8,4) probe-top FULL orchestrator reduction, OFF vs ON (greedy sector symmetry),
# in parallel, recommended config (pentagonbox_nosym + --paper-masters-only).
#
# Magnitude test: does the inference-only symmetry pre-reduction drastically cut
# the number of DISTINCT integrals (= Condor workers) needed to reduce a hard top
# integral to the unique 62-master IBP+LI basis?  The (8,4) top is the heaviest of
# the four benchmark probes short of the 21.6 h (8,5) production run (its one-step
# worker alone is ~3,400 s).
#
# Deliverables (compare_symgate.py): masters MUST be identical (correctness gate);
# worker counts OFF vs ON (the headline); ON symmetry firings; wall-clock (note:
# the two runs share the cluster, so wall-clock is contended — worker count is the
# clean, contention-free metric).
# ============================================================================
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
RED=$BASE/reduction
INT='-1,2,1,0,1,2,1,1,-3,0,0'        # (8,4) probe top

echo "=== launching OFF  -> results/probetop84_off ==="
bash "$RED/run_reduction.sh" "$INT" "$BASE/results/probetop84_off"
sleep 3
echo "=== launching ON (SAILIR_SYMMETRY=1) -> results/probetop84_on ==="
SAILIR_SYMMETRY=1 bash "$RED/run_reduction.sh" "$INT" "$BASE/results/probetop84_on"
sleep 6
echo
echo "=== live orchestrator processes (hierarchical_reduction.py) ==="
ps -u dshih -o pid=,ppid=,etime=,cmd= | grep -F "hierarchical_reduction.py" | grep -v grep
echo
echo "logs:"
echo "  OFF: $BASE/results/probetop84_off/logs/hierarchical.log"
echo "  ON : $BASE/results/probetop84_on/logs/hierarchical.log"
