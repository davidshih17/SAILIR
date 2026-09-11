#!/usr/bin/env python
"""Inter-iteration timing distribution from an orchestrator log's t= stamps.

Usage: iter_timing_dist.py <orch_log> [threshold_s]
Prints the distribution and every gap >= threshold (default 30s).
"""
import re
import sys
from datetime import datetime

log = sys.argv[1]
thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
PAT = re.compile(r"\[Iter (\d+)\].*t=(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

prev = None
gaps = []
slow = []
for raw in open(log, errors="replace"):
    for ln in raw.split("\r"):
        m = PAT.search(ln)
        if not m:
            continue
        it = int(m.group(1))
        ts = datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S").timestamp()
        if prev is not None and it == prev[0] + 1:
            d = ts - prev[1]
            gaps.append(d)
            if d >= thresh:
                slow.append((it, d))
        prev = (it, ts)

gaps.sort()
n = len(gaps)
if not n:
    print("no consecutive iteration stamps found")
    sys.exit(0)
print(f"{n} iteration gaps: med={gaps[n//2]:.1f}s p90={gaps[int(n*.9)]:.1f}s "
      f"p99={gaps[int(n*.99)]:.1f}s max={gaps[-1]:.1f}s "
      f"total={sum(gaps)/60:.1f}min")
print(f"\ngaps >= {thresh:.0f}s: {len(slow)}")
for it, d in slow:
    print(f"  iter {it}: {d:.0f}s")
