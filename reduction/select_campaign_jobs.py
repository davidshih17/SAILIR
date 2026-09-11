#!/usr/bin/env python
"""Select ONLY this campaign's worker jobs from the Condor queue.

The queue holds unrelated work of the user's alongside the campaign. Selection
therefore requires BOTH `greedy_worker.py` and the campaign tag in the job's
arguments -- either alone is not sufficient.

A cluster that mixes campaign and non-campaign jobs must never be removed whole:
those are emitted as individual cluster.proc ids instead. Whole clusters are
emitted only when every job in them is a campaign job, which keeps the command
line short for the 1,900-job batch clusters.
"""
import subprocess
import sys
from collections import defaultdict


def main():
    tag = sys.argv[1]
    count_only = '--count-only' in sys.argv
    out = subprocess.run(['condor_q', '-af', 'ClusterId', 'ProcId', 'Args'],
                         capture_output=True, text=True).stdout

    per_cluster = defaultdict(lambda: [0, 0])     # cluster -> [campaign, other]
    campaign_procs = defaultdict(list)
    n_campaign = n_other = 0
    for line in out.splitlines():
        f = line.split(None, 2)
        if len(f) < 2:
            continue
        cid, pid = f[0], f[1]
        args = f[2] if len(f) > 2 else ''
        is_campaign = ('greedy_worker.py' in args) and (tag in args)
        per_cluster[cid][0 if is_campaign else 1] += 1
        if is_campaign:
            campaign_procs[cid].append(pid)
            n_campaign += 1
        else:
            n_other += 1

    if count_only:
        print(f'campaign worker jobs still queued: {n_campaign:,}')
        print(f'other jobs in the queue (untouched): {n_other:,}')
        return

    whole, partial, mixed = [], [], 0
    for cid, (n_c, n_o) in per_cluster.items():
        if not n_c:
            continue
        if n_o:
            mixed += 1
            partial.extend(f'{cid}.{p}' for p in campaign_procs[cid])
        else:
            whole.append(cid)

    print(f'# campaign jobs: {n_campaign:,}   other jobs (untouched): {n_other:,}')
    print(f'# whole clusters: {len(whole)}   MIXED clusters removed per-job: '
          f'{mixed} ({len(partial)} jobs)')
    for cid in sorted(whole, key=int):
        print(cid)
    for jid in sorted(partial):
        print(jid)


if __name__ == '__main__':
    main()
