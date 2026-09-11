#!/bin/bash
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
for pair in "2,1,1,0,1,1,0,0,0,0,0:A" "1,1,2,0,1,1,0,0,0,0,0:B"; do
  INT="${pair%:*}"; TAG="${pair#*:}"
  bash $BASE/reduction/launch_corner.sh "$INT" "ph_$TAG"
done
