"""Build ONE system, emit closures for every target that shares it.

The per-target job design rebuilt the same system ~8 times over; this builds it
once in memory (SAILIR_KEEP_SYSTEMS=1) and never writes it to disk
(SAILIR_NO_SYSTEM_CACHE=1) -- the systems dir is already 101 GB with single
systems up to 1.47 GB on a filesystem at 91%, and with grouping the on-disk
cache buys nothing.

Emits, per target:
  out/<tag>.json     the closure library: the (op, seed) SET, same schema and
                     same tag convention as the per-target worker, so downstream
                     consumers are unchanged
  status.jsonl       one row per target: ok/failed, elapsed, closure size.
                     This is where the TIMING DISTRIBUTION comes from -- the
                     thing we actually need to size the campaign -- measured on
                     all 18,000 rather than extrapolated from a handful.

Restartable: a target whose closure already exists is skipped, so a job that
dies partway resumes rather than redoing its build.
"""
import argparse
import json
import os
import resource
import sys
import threading
import time

B = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
os.environ.setdefault('SAILIR_TOPOLOGY', 'gravity3L')
os.environ.setdefault('SAILIR_SECTOR_RANK', '1')
# Grouping is the whole point: keep the built system in RAM across targets and
# never pickle it.
os.environ['SAILIR_KEEP_SYSTEMS'] = '1'
os.environ['SAILIR_NO_SYSTEM_CACHE'] = '1'
sys.path.insert(0, B)
sys.path.insert(0, f'{B}/reduction')


def tag_of(t):
    # MUST match worker.sh / eqact_dataset.tag_of: commas -> underscores, and
    # NO '-' stripping (that collided 1,377 of 18,000 once indices go negative).
    return '_'.join(str(int(x)) for x in t)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def rss_now_mb():
    """Live RSS. ru_maxrss is a high-water mark and only moves at coarse
    granularity, so the watchdog reads /proc directly."""
    with open('/proc/self/statm') as fh:
        return int(fh.read().split()[1]) * (os.sysconf('SC_PAGE_SIZE') / 2**20)


def start_memory_watchdog(cap_mb, marker, gid):
    """Abort the job if it approaches its memory grant.

    Memory here is NOT predictable in advance: at a fixed seed count, observed
    peak RSS spanned 3-4x (101,640 seeds gave anywhere from 1.2 GB to 4.0 GB),
    because cost depends on how many rules a sector yields and how long their
    tails are, not on the box dimensions. So the bound has to be enforced at
    RUNTIME rather than guessed when sizing the request.

    The breach happens INSIDE build_system, a single long call, so no
    Python-level check between targets would ever fire. A daemon thread samples
    /proc and hard-exits, leaving a marker so the group can be retried at the
    next CPU tier (each tier keeps 10 GB per CPU).

    os._exit is deliberate: a normal exception would be caught by the per-target
    handler and the job would sail on, still over its grant.
    """
    def _w():
        while True:
            try:
                cur = rss_now_mb()
                if cur > cap_mb:
                    with open(marker, 'w') as f:
                        json.dump({'group': gid, 'rss_mb': round(cur),
                                   'cap_mb': cap_mb}, f)
                    sys.stderr.write(
                        f'[{gid}] MEMORY CAP: {cur:.0f} MB > {cap_mb} MB '
                        f'-- aborting for retry at a higher CPU tier\n')
                    sys.stderr.flush()
                    os._exit(42)
            except Exception:
                pass
            time.sleep(1.0)
    threading.Thread(target=_w, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--dr', type=int, default=1)
    ap.add_argument('--ds', type=int, default=1)
    ap.add_argument('--budget', type=int, default=500000)
    ap.add_argument('--cap-mb', type=int, default=9000,
                    help='abort above this RSS; keep under the per-CPU grant')
    ap.add_argument('--retry-dir', default=None,
                    help='where to drop a marker when the cap trips')
    args = ap.parse_args()

    os.environ['SAILIR_SEED_BUDGET'] = str(args.budget)
    with open(args.group) as f:
        g = json.load(f)
    targets = [tuple(t) for t in g['targets']]
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'out'), exist_ok=True)

    gid = os.path.splitext(os.path.basename(args.group))[0]
    print(f'[{gid}] sector={g["sector"]} r<={g["rmax"]} s<={g["smax"]} '
          f'seeds={g["seeds"]:,} targets={len(targets)} '
          f'cap={args.cap_mb}MB', flush=True)
    rdir = args.retry_dir or os.path.join(args.outdir, 'retry')
    os.makedirs(rdir, exist_ok=True)
    start_memory_watchdog(args.cap_mb, os.path.join(rdir, f'{gid}.json'), gid)

    import topo_config as _tc
    from truth_engine import TruthEngine

    t_boot = time.time()
    eng = TruthEngine(_tc.TOPO_DIR, os.path.join(
        _tc.TOPO_DIR, 'kira_validate/sectormappings/GR/trivialsector'))
    print(f'[{gid}] engine up in {time.time()-t_boot:.1f}s', flush=True)

    status = open(os.path.join(args.outdir, f'status_{gid}.jsonl'), 'a')
    n_ok = n_fail = n_skip = 0
    t_group = time.time()
    for i, T in enumerate(targets):
        tag = tag_of(T)
        dest = os.path.join(args.outdir, 'out', f'{tag}.json')
        if os.path.exists(dest):
            n_skip += 1
            continue
        t0 = time.time()
        try:
            wr = eng.worker_replay(T, dr=args.dr, ds=args.ds, verbose=False)
            closure = [(op, tuple(x + d for x, d in zip(J, delta)))
                       for (J, op, delta) in wr['actions']]
            # PIVOTS, parallel to `closure`. worker_replay knows J but the
            # original dump threw it away: the closure is meant to be the
            # STATE-INDEPENDENT set of (op, seed) rows for the beam's candidate
            # restriction, and the pivot is state-dependent (the same equation
            # eliminates a different integral depending on the substitution
            # store). True for that consumer, but it makes diagnostics
            # re-derive what the truth already knew. Kept as a SEPARATE key so
            # `for op, seed in d['closure']` still works; costs ~77 MB across
            # all 18,000 (0.086 -> 0.163 GB), which is free at 1.9 TB.
            pivots = [list(J) for (J, _op, _d) in wr['actions']]
            tmp = dest + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'integral': list(T), 'sector': g['sector'],
                           'closure': [[int(o), list(s)] for o, s in closure],
                           'pivots': pivots}, f)
            os.replace(tmp, dest)          # atomic: no half-written closure
            dt = time.time() - t0
            n_ok += 1
            rec = {'tag': tag, 'ok': True, 'sec': round(dt, 2),
                   'rows': len(closure), 'rss_mb': round(rss_mb())}
        except Exception as e:
            dt = time.time() - t0
            n_fail += 1
            rec = {'tag': tag, 'ok': False, 'sec': round(dt, 2),
                   'err': f'{type(e).__name__}: {e}'[:300]}
        status.write(json.dumps(rec) + '\n')
        status.flush()
        print(f'[{gid}] {i+1}/{len(targets)} {"OK " if rec["ok"] else "FAIL"} '
              f'{rec["sec"]}s rss={rss_mb():.0f}MB', flush=True)
    status.close()
    print(f'[{gid}] DONE ok={n_ok} fail={n_fail} skip={n_skip} '
          f'group_time={time.time()-t_group:.1f}s peak_rss={rss_mb():.0f}MB',
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
