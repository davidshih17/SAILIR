"""Kill DIVERGING orchestrator workers and record them for study.

Two independent kill criteria, either one is sufficient:
  * NON-MASTER SPIRAL  nm >= --nm-min (default 500)
  * MEMORY SPIRAL      MemoryUsage >= --mem-max-mb (default 20480 = 20 GB)
Note the orchestrator reserves only --worker-memory-gb (4 GB) per worker, so a
job anywhere near the memory ceiling is already far over its reservation.

A diverging worker never finishes: its non-master count and substitution store
grow without bound (measured on g1023: nm 2261 from a start of 3, weight r=215
from a start of 11, store 592k entries). With no wall cap on orchestrator
workers it holds a CPU forever.

SAFE TO KILL EXTERNALLY -- verified in hierarchical_reduction.py:
  * the reaper (line ~1244) acts only when the output file EXISTS, so a killed
    worker's entry simply stays in `pending` and is never reaped;
  * there is NO missing-cluster resubmit path;
  * the straggler path is the only resubmitter and it is DISABLED whenever
    --straggler-timeout exceeds 1e8 (production sets 1e9).
  => a killed worker is killed ONCE. No kill/resubmit loop is possible.

COST: `available_slots = max_concurrent - len(pending)` (line ~1006), so the
dead entry still consumes an orchestrator submission slot. The physical CPU is
freed; the orchestrator's concurrency budget is not. The integral itself will
never resolve, so the reduction cannot COMPLETE while any killed integral is
still required -- acceptable only when the run is being used for study.

--dry-run prints what it would kill and touches nothing.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

LINE = re.compile(r'\[v6 step +(\d+)\].*?best mw=\((\d+), (\d+), \(([^)]*)\)\).*?'
                  r'nm=(\d+).*?rs_vsz=(\d+).*?t_total=([0-9.]+)')
SUBMIT = re.compile(r'^000 \((\d+)\.(\d+)\.\d+\)')


def memory_by_cluster():
    """{'cluster.proc': MemoryUsage_MB} for everything currently in the queue.

    ONE condor_q per sweep, not one per worker -- a per-worker query would be
    thousands of schedd round-trips and is how the schedd got jammed earlier.
    """
    try:
        r = subprocess.run(
            ['condor_q', '-af:,', 'ClusterId', 'ProcId', 'MemoryUsage'],
            capture_output=True, text=True, timeout=120)
    except Exception:
        return {}
    out = {}
    for line in r.stdout.splitlines():
        f = [x.strip() for x in line.split(',')]
        if len(f) < 3:
            continue
        try:
            out[f'{f[0]}.{f[1]}'] = int(float(f[2]))
        except ValueError:
            pass   # MemoryUsage is 'undefined' until the job has run a while
    return out


def cluster_of(logdir, base):
    """Condor cluster.proc for a worker, from its own .log submit event."""
    p = os.path.join(logdir, base + '.log')
    if not os.path.exists(p):
        return None
    for line in open(p, errors='ignore'):
        m = SUBMIT.match(line)
        if m:
            return f'{m.group(1)}.{m.group(2)}'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--nm-min', type=int, default=500)
    ap.add_argument('--mem-max-mb', type=int, default=20480)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    logdir = os.path.join(a.run_dir, 'work/logs')
    killed_db = os.path.join(a.run_dir, 'killed_diverging.jsonl')
    already = set()
    if os.path.exists(killed_db):
        for l in open(killed_db):
            try:
                already.add(json.loads(l)['log'])
            except Exception:
                pass

    mem = memory_by_cluster()
    victims = []
    for f in glob.glob(os.path.join(logdir, '*.out')):
        s = open(f, errors='ignore').read()
        if 'SUCCESS in' in s or 'INCOMPLETE in' in s:
            continue
        ms = LINE.findall(s)
        if not ms:
            continue
        base = os.path.basename(f)[:-4]
        if base in already:
            continue
        cl = cluster_of(logdir, base)
        mb = mem.get(cl, 0) if cl else 0
        step, r, sN, _, nm, rsv, t = ms[-1]
        reasons = []
        if int(nm) >= a.nm_min:
            reasons.append(f'nm>={a.nm_min}')
        if mb >= a.mem_max_mb:
            reasons.append(f'mem>={a.mem_max_mb}MB')
        if not reasons:
            continue
        victims.append(dict(log=base, cluster=cl, reason='+'.join(reasons),
                            mem_mb=mb,
                            nm=int(nm), nm_start=int(ms[0][4]),
                            max_w_r=int(r), max_w_r_start=int(ms[0][1]),
                            step=int(step), rs_vsz=int(rsv),
                            t_total=float(t), killed_at=time.strftime('%F %T')))
    if not victims:
        print('no new diverging workers')
        return 0
    ids = [v['cluster'] for v in victims if v['cluster']]
    n_mem = sum(1 for v in victims if 'mem' in v['reason'])
    print(f'{len(victims)} diverging (nm>={a.nm_min} or mem>={a.mem_max_mb}MB); '
          f'{n_mem} on MEMORY; {len(ids)} with a cluster id')
    for v in victims[:10]:
        print(f'  {v["log"][:46]:48s} nm={v["nm"]:>5} mem={v["mem_mb"]:>6}MB '
              f'w_r={v["max_w_r"]:>4} t={v["t_total"]:.0f}s [{v["reason"]}]')
    if a.dry_run:
        print('\nDRY RUN — nothing killed')
        return 0
    if ids:
        subprocess.run(['condor_rm'] + ids, capture_output=True, text=True,
                       timeout=180)
    with open(killed_db, 'a') as fh:
        for v in victims:
            fh.write(json.dumps(v) + '\n')
    print(f'\nkilled {len(ids)}; recorded {len(victims)} -> {killed_db}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
