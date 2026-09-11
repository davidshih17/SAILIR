#!/usr/bin/env python
"""Spot-by-spot comparison of the two g1023 bench variants at the production
slow-iteration stream ranges (slow_iter_watchlist.txt).

For each bench log: reconstruct (stream_position, wall_time) from the
periodic '[bench iter N] ... stream S/...' lines (file offsets give
ordering; wall time from cumulative it= is unreliable, so we use the line
count of expensive events instead) — and attribute to each watchlist range
the [route-cascade] lines and worst iteration times printed while the
bench's stream counter was inside the range.
"""
import os
import re
import sys

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
B = os.path.join(ROOT, "results/orch_bench")

wl = []
for ln in open(os.path.join(B, "slow_iter_watchlist.txt")):
    if ln.startswith("#"):
        continue
    p = ln.split()
    wl.append((p[0], int(p[1]), float(p[2]), int(p[5]), int(p[6])))
# merge overlapping/adjacent ranges, keep total production duration
wl.sort(key=lambda r: r[3])
merged = []
for log, it, dur, i0, i1 in wl:
    if merged and i0 <= merged[-1][1] + 200:
        m = merged[-1]
        merged[-1] = (m[0], max(m[1], i1), m[2] + dur)
    else:
        merged.append((i0, i1, dur))

ITER = re.compile(r"\[bench iter (\d+)\] .*stream (\d+)/\d+ .*it=([0-9.]+)s")
CASC = re.compile(r"\[route-cascade\] (\d+) rounds, ([0-9.]+)s, (\d+) routed")


def parse(path):
    events = []          # (stream_pos_estimate, kind, value)
    cur_stream = 0
    for ln in open(path, errors="replace"):
        m = ITER.search(ln)
        if m:
            cur_stream = int(m.group(2))
            events.append((cur_stream, "it", float(m.group(3))))
            continue
        c = CASC.search(ln)
        if c:
            events.append((cur_stream, "casc",
                           (int(c.group(1)), float(c.group(2)),
                            int(c.group(3)))))
    return events


base = parse(os.path.join(B, "base2_g1023 0.out"))
incr = parse(os.path.join(B, "incr_g1023 1.out"))
base_max = max(s for s, _, _ in base)
incr_max = max(s for s, _, _ in incr)
print(f"bench coverage: base to stream {base_max}, incr to {incr_max}\n")

print(f"{'stream range':>17} {'prod_slow':>9} | {'BASE casc(rounds/s)':>24} "
      f"{'maxit':>6} | {'INCR casc(rounds/s)':>24} {'maxit':>6}")
for i0, i1, dur in merged:
    if i0 > min(base_max, incr_max):
        continue
    def collect(ev):
        cascs = [(v[0], v[1]) for s, k, v in ev if k == "casc"
                 and i0 - 400 <= s <= i1 + 400]
        its = [v for s, k, v in ev if k == "it" and i0 - 400 <= s <= i1 + 400]
        cs = (f"{sum(c[0] for c in cascs)}r/{sum(c[1] for c in cascs):.0f}s"
              f"({len(cascs)}x)") if cascs else "-"
        return cs, (max(its) if its else 0)
    bc, bi = collect(base)
    ic, ii = collect(incr)
    print(f"{i0:>8}-{i1:<8} {dur/60:>7.1f}m | {bc:>24} {bi:>6.1f} | "
          f"{ic:>24} {ii:>6.1f}")
