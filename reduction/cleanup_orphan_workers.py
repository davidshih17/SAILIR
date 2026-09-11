#!/usr/bin/env python
"""Targeted removal of ORPHANED g1023 worker jobs (generations dispatched by
killed orchestrators). Never touches the live orchestrator's own jobs (its
pending table must stay intact — straggler resubmission is disabled).

An orphan job is removed iff:
  (a) its integral's result is already banked in work/results, OR
  (b) the live orchestrator (jobs queued at/after its start time) has its own
      job for the same integral, OR
  (c) it duplicates an older orphan on the same integral (keep the oldest,
      which has the most compute invested).

Usage: cleanup_orphan_workers.py <live_orch_pid> <work_results_dir> [--apply]
"""
import os
import subprocess
import sys

pid = sys.argv[1]
resdir = sys.argv[2]
apply_mode = "--apply" in sys.argv

# live orchestrator start epoch
out = subprocess.run(["ps", "-o", "lstart=", "-p", pid],
                     capture_output=True, text=True).stdout.strip()
t0 = int(subprocess.run(["date", "-d", out, "+%s"],
                        capture_output=True, text=True).stdout.strip())

q = subprocess.run(["condor_q", "dshih", "-af", "ClusterId", "ProcId",
                    "QDate", "Args"], capture_output=True, text=True).stdout
jobs = []
for ln in q.splitlines():
    parts = ln.split(None, 3)
    if len(parts) < 4 or "--integral=" not in parts[3]:
        continue
    integ = parts[3].split("--integral='")[1].split("'")[0]
    jobs.append((f"{parts[0]}.{parts[1]}", int(parts[2]), integ))

live = {}
orphans = []
for jid, qd, integ in jobs:
    if qd >= t0:
        live.setdefault(integ, []).append(jid)
    else:
        orphans.append((jid, qd, integ))

banked = set()
for integ in {i for _, _, i in orphans}:
    f = integ.replace(",", "_")
    import glob
    if glob.glob(os.path.join(resdir, f"async_*_{f}.pkl")):
        banked.add(integ)

rm = []
keep_oldest = {}
for jid, qd, integ in sorted(orphans, key=lambda x: x[1]):
    if integ in banked:
        rm.append((jid, "banked"))
    elif integ in live:
        rm.append((jid, "live-dup"))
    elif integ in keep_oldest:
        rm.append((jid, "orphan-dup"))
    else:
        keep_oldest[integ] = jid

from collections import Counter
c = Counter(reason for _, reason in rm)
print(f"jobs total: {len(jobs)}  live(orch-owned): "
      f"{sum(len(v) for v in live.values())}  orphans: {len(orphans)}")
print(f"orphans kept (unique, unbanked, not covered live): {len(keep_oldest)}")
print(f"to remove: {len(rm)}  breakdown: {dict(c)}")
if not apply_mode:
    print("(report only; --apply to remove)")
    sys.exit(0)
ids = [jid for jid, _ in rm]
for i in range(0, len(ids), 200):
    subprocess.run(["condor_rm"] + ids[i:i + 200], capture_output=True)
print(f"removed {len(ids)} jobs")
