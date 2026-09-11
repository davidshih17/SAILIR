#!/bin/bash
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
i=0
while IFS= read -r line; do
    INT=$(python3 -c "import sys,re;print(re.match(r'TA\[(.*)\]',sys.argv[1]).group(1))" "$line")
    bash $BASE/reduction/launch_corner.sh "$INT" $i
    i=$((i+1))
done < $BASE/reduction/uncovered_corners.txt
echo "launched $i corner reductions"
