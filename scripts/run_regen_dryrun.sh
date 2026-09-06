#!/bin/bash
# Validate run_closure_regen.sh as far as this box allows: the NEW/USED split
# and make_groups both run fully; batch_closure is expected to fail on the
# missing gravity3L engine pkls, and MUST do so without aborting the loop.
set -uo pipefail
cd /home/shih/work/SAILIR_p101
./reduction/run_closure_regen.sh results/truth/closures/regen/missing_targets.txt
echo "=== exit=$? ==="
