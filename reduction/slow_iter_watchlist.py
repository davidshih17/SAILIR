#!/usr/bin/env python
"""SLOW-ITERATION WATCHLIST: mine the production logs for every iteration
slower than a threshold, and map each to its position in the recorded arrival
stream (result-file mtimes) so the replay benches can be examined at exactly
those spots.

Output: results/orch_bench/slow_iter_watchlist.txt with one line per slow
iteration: log, iter, duration_s, wall window, stream index range, #arrivals.
"""
import glob
import os
import re
import sys
from datetime import datetime

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
G = os.path.join(ROOT, "results/gr_reduce/g1023")
THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0

# arrival stream index by mtime (same ordering as the bench)
files = glob.glob(os.path.join(G, "work/results/*.pkl"))
arr = sorted(os.path.getmtime(f) for f in files)
print(f"stream: {len(arr)} results")


def stream_index(ts):
    import bisect
    return bisect.bisect_right(arr, ts)


PAT = re.compile(r"\[Iter (\d+)\].*t=(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
rows = []
for log in sorted(glob.glob(os.path.join(G, "logs/orch_v*.log"))):
    prev = None
    for raw in open(log, errors="replace"):
        for ln in raw.split("\r"):
            m = PAT.search(ln)
            if not m:
                continue
            it = int(m.group(1))
            ts = datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S").timestamp()
            if prev is not None and it == prev[0] + 1:
                dur = ts - prev[1]
                if dur >= THRESH:
                    i0, i1 = stream_index(prev[1]), stream_index(ts)
                    rows.append((os.path.basename(log), it, dur,
                                 prev[1], ts, i0, i1))
            prev = (it, ts)

rows.sort(key=lambda r: -r[2])
out = os.path.join(ROOT, "results/orch_bench/slow_iter_watchlist.txt")
with open(out, "w") as f:
    f.write("# log iter dur_s wall_start wall_end stream_i0 stream_i1 arrivals\n")
    for log, it, dur, t0, t1, i0, i1 in rows:
        f.write(f"{log} {it} {dur:.0f} "
                f"{datetime.fromtimestamp(t0):%m-%d_%H:%M:%S} "
                f"{datetime.fromtimestamp(t1):%m-%d_%H:%M:%S} "
                f"{i0} {i1} {i1 - i0}\n")
print(f"{len(rows)} iterations >= {THRESH:.0f}s  -> {out}")
print("\nworst 15:")
for log, it, dur, t0, t1, i0, i1 in rows[:15]:
    print(f"  {log} iter {it}: {dur/60:.1f} min, stream {i0}-{i1} "
          f"({i1 - i0} arrivals)")
