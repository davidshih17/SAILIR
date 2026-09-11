#!/bin/bash
# Detached resume launcher. Absolute paths only, and no reliance on the caller's
# cwd or process group: a launch tied to the calling tool call dies silently with
# it (launch4 produced a zero-byte log and no orchestrator).
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
export RESUME=1
export SAILIR_BOTTOM_UP=1
export SAILIR_RECIPE_INDEX=$B/results/recipe_index_full.npz
cd "$B" || exit 1
exec bash "$B/reduction/run_orch_greedy_certified.sh" \
    1,1,1,1,1,1,1,1,2,2,0,0,0,0,0 g1023_greedy_certified
