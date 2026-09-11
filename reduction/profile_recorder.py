#!/usr/bin/env python
"""Profile the truth recorder's per-step cost to find where enum time goes.

Runs truth_record_training_v2 under cProfile with a sample cap, and separately
reports per-step wall time so superlinear growth (if any) is visible.
Usage: SAILIR_TOPOLOGY=gravity3L python profile_recorder.py <integral> <cap>
"""
import cProfile
import io
import os
import pstats
import sys

sys.argv = sys.argv[:]
integral = sys.argv[1] if len(sys.argv) > 1 else "2,2,1,1,1,1,0,0,0,0,0,0,0,0,0"
cap = sys.argv[2] if len(sys.argv) > 2 else "40"
os.environ["SAILIR_MAX_SAMPLES"] = cap
os.environ.setdefault("SAILIR_TOPOLOGY", "gravity3L")
os.environ.setdefault("SAILIR_SECTOR_RANK", "1")
os.environ["SAILIR_STEP_LOG"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data-gen"))

import truth_record_training_v2 as rec

out = f"/tmp/prof_rec_{integral.replace(',', '_')}.jsonl"
argv = ["truth_record_training_v2.py", "--integral", integral, "--output", out]


def run():
    sys.argv = argv
    try:
        rec.main()
    except SystemExit:
        pass


pr = cProfile.Profile()
pr.enable()
run()
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(28)
print("\n================ CUMULATIVE (top 28) ================")
print(s.getvalue())

s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
ps2.print_stats(20)
print("\n================ TOTTIME (top 20, self-time) ================")
print(s2.getvalue())
