#!/bin/bash
# Regression A/B AFTER the 139/canon strip: rerun the four successful symmetry
# inferences (gate_small, m1, m2, m3) baseline vs --use-symmetry. Confirms the strip
# left the 174 routing intact: masters must still match (gate PASS) and the savings
# must reproduce. New *_poststrip tags — does NOT overwrite the prior run dirs.
set -e
D=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/reduction

bash $D/ab_run.sh "1,-1,1,0,1,1,0,0,0,0,0"      gate_small_poststrip
bash $D/ab_run.sh "1,1,-1,1,1,1,-1,1,-1,0,0"    m1_poststrip
bash $D/ab_run.sh "2,0,1,0,1,-1,0,1,-1,-1,0"    m2_poststrip
bash $D/ab_run.sh "1,1,0,1,1,0,-3,1,0,0,0"      m3_poststrip

echo "all four A/B pairs launched (8 orchestrators)."
