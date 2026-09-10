#!/usr/bin/env python
"""Condor dispatch of symmetry-routing closure batches (2026-07-30).

route_batch_condor(candidates, work_dir, tag): writes batches of 100
integrals, submits one Condor cluster of routing_closure_worker.py jobs
(2 GB / 1 cpu each), polls for the output pkls, merges and returns
{integral: composed_rule_or_None}. Batches whose job vanishes without
output are recomputed serially in-process (fallback, never lost)."""
import os
import glob
import pickle
import subprocess
import time

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
PYTHON = "/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python"
BATCH = 100


def route_batch_condor(candidates, work_dir, tag, poll_s=10, timeout_s=21600):
    """Route `candidates` on Condor in 100-integral batches.

    TIMEOUT SIZING. On expiry every batch without an output file is recomputed
    SERIALLY IN-PROCESS, which is not a graceful degradation at this scale:
    measured routing cost is ~10s/integral early in a batch and ~27s once the
    per-worker cache has grown, so a single missing 100-integral batch is ~45
    minutes of blocking serial work and 10% of 1,044 batches is ~100 hours.
    The old 3600s default was set when batches were small; a 104,383-integral
    pass needs hours, so the deadline must not be the thing that decides.
    """
    rdir = os.path.join(work_dir, "routing")
    for sub in ("batches", "out", "logs"):
        os.makedirs(os.path.join(rdir, sub), exist_ok=True)
    wsh = os.path.join(rdir, "worker.sh")
    if not os.path.exists(wsh):
        with open(wsh, "w") as f:
            # PROPAGATE THE FULL SYMMETRY ENVIRONMENT. Reading os.environ and
            # defaulting is how 7,144 jobs died at once: the orchestrator
            # process had no SAILIR_SECTOR_RANK, so this wrote
            # SAILIR_SECTOR_RANK=0 and every worker hit
            #   "staged routing requires SAILIR_SECTOR_RANK=1".
            # Worse would have been SILENT: without SAILIR_SYM_PRIME /
            # SAILIR_SYM_STORE the workers load the p=1009 transform store
            # while the orchestrator runs at p=101, and rules come back in the
            # WRONG FIELD with no error at all. Defaults are the values these
            # workers actually require, not neutral ones.
            _env = {
                'SAILIR_TOPOLOGY':   os.environ.get('SAILIR_TOPOLOGY', ''),
                'SAILIR_SECTOR_RANK': os.environ.get('SAILIR_SECTOR_RANK', '1'),
            }
            for _k in ('SAILIR_SYM_PRIME', 'SAILIR_SYM_Y', 'SAILIR_SYM_STORE'):
                if os.environ.get(_k):
                    _env[_k] = os.environ[_k]
            if _env['SAILIR_SECTOR_RANK'] != '1':
                raise SystemExit(
                    'route_batch_condor: SAILIR_SECTOR_RANK must be 1 for '
                    'staged/canonical routing -- the workers assert it and '
                    'would all die on arrival.')
            _exports = ' '.join(f'{k}={v}' for k, v in _env.items())
            f.write(
                "#!/bin/bash\n"
                f"export {_exports}\n"
                f"PYTHONUNBUFFERED=1 {PYTHON} "
                f"{ROOT}/reduction/routing_closure_worker.py \"$1\" \"$2\"\n")
        os.chmod(wsh, 0o755)

    cand = list(candidates)
    batches = []
    for i in range(0, len(cand), BATCH):
        bf = os.path.join(rdir, "batches", f"rt_{tag}_{i // BATCH}.txt")
        of = os.path.join(rdir, "out", f"rt_{tag}_{i // BATCH}.pkl")
        with open(bf, "w") as f:
            for t in cand[i:i + BATCH]:
                f.write(",".join(str(x) for x in t) + "\n")
        batches.append((bf, of))

    listf = os.path.join(rdir, f"rt_{tag}.list")
    with open(listf, "w") as f:
        for bf, of in batches:
            f.write(f"{bf} {of}\n")
    subf = os.path.join(rdir, f"rt_{tag}.sub")
    with open(subf, "w") as f:
        f.write(f"""universe = vanilla
executable = {wsh}
arguments = "$(bf) $(of)"
request_cpus = 1
request_memory = 2GB
request_disk = 1GB
log = {rdir}/logs/rt_{tag}.log
output = {rdir}/logs/rt_{tag}_$(Process).out
error = {rdir}/logs/rt_{tag}_$(Process).err
queue bf,of from {listf}
""")
    r = subprocess.run(["condor_submit", subf], capture_output=True, text=True,
                       timeout=120)
    print(f"[route-condor] submitted {len(batches)} batches "
          f"({len(cand)} integrals): {r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[:100]}",
          flush=True)

    t0 = time.time()
    cluster = None
    for ln in (r.stdout or "").splitlines():
        if "submitted to cluster" in ln:
            cluster = ln.split("cluster")[1].strip().rstrip(".")
    while time.time() - t0 < timeout_s:
        missing = [of for _, of in batches if not os.path.exists(of)]
        if not missing:
            break
        alive = 1
        if cluster:
            q = subprocess.run(["condor_q", cluster, "-af", "ProcId"],
                               capture_output=True, text=True, timeout=60)
            alive = len(q.stdout.split())
        if alive == 0 and missing:
            print(f"[route-condor] {len(missing)} batch outputs missing with "
                  f"no jobs left — computing those serially", flush=True)
            break
        time.sleep(poll_s)

    out = {}
    serial_fallback = []
    for bf, of in batches:
        if os.path.exists(of):
            try:
                with open(of, "rb") as f:
                    out.update(pickle.load(f))
                continue
            except Exception:
                pass
        serial_fallback.append(bf)
    if serial_fallback:
        print(f"[route-condor] WARNING: {len(serial_fallback)} batch(es) had no "
              f"output after {time.time()-t0:.0f}s — recomputing "
              f"~{len(serial_fallback)*BATCH} integrals SERIALLY in-process. "
              f"At ~27s each this is ~{len(serial_fallback)*BATCH*27/3600:.1f}h "
              f"and the orchestrator is BLOCKED throughout.", flush=True)
        import sys
        sys.path.insert(0, os.path.join(ROOT, "reduction"))
        from symmetry_route import canonical_monolithic_rule as _route
        for bf in serial_fallback:
            for ln in open(bf):
                ln = ln.strip()
                if ln:
                    t = tuple(int(x) for x in ln.split(","))
                    out[t] = _route(t)   # raw (uncomposed) rule — still exact
    # tidy batch inputs; keep outputs until next tag reuses names
    for bf, _ in batches:
        try:
            os.remove(bf)
        except OSError:
            pass
    n_r = sum(1 for v in out.values() if v is not None)
    print(f"[route-condor] merged {len(out)} routings ({n_r} routable) in "
          f"{time.time() - t0:.0f}s", flush=True)
    return out
