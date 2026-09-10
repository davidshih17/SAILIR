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
# INTEGRALS PER CONDOR JOB. 1 = one job per integral, which is the point:
# routing cost is dominated by a small pathological tail, and at BATCH=100 a
# single slow integral blocks the 99 healthy ones sharing its job AND withholds
# their results, because the worker writes its pkl only at the end. Measured on
# the live g1023 pass: 40 minutes in, 1,044 batches had produced 2 outputs while
# ~16,000 integrals were already routed but unwritten.
#
# Startup does NOT argue against this: the worker needs only symmetry_route and
# a 200 KB transform store, never the IBP topology -- measured import+first-call
# overhead is 0.02s, so per-integral jobs waste no meaningful setup.
#
# The real constraint is the schedd, not the worker: 104k jobs submitted at once
# would jam it (PRODUCTION_DATA_GENERATION.md PITFALL 11 -- 15,000 jobs left it
# unable to answer condor_q OR condor_rm). Hence SUBMIT_CHUNK drip feeding.
BATCH = int(os.environ.get('SAILIR_ROUTE_BATCH', '100'))
SUBMIT_CHUNK = int(os.environ.get('SAILIR_ROUTE_SUBMIT_CHUNK', '5000'))
QUEUE_CEILING = int(os.environ.get('SAILIR_ROUTE_QUEUE_CEILING', '8000'))


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
    # ALWAYS REWRITE. This used to be `if not os.path.exists(wsh)`, so the file
    # from a previous run survived and every later change to the propagated
    # environment was silently ignored -- the jobs kept launching with the old
    # exports while the source said otherwise.
    if True:
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
            # PROPAGATE THE ROUTING CAPS EXPLICITLY, never by default.
            # SAILIR_ROUTE_MAX_EXPAND is the only cap that acts DURING the
            # multinomial expansion; MAX_TERMS can only reject a rule once it is
            # already built, and MAX_S only screens the top integral, not the
            # intermediates canonical_monolithic_rule generates internally.
            # Omitting it means canonicalize2 reads its '0' default = NO LIMIT.
            # Measured cost of that omission on the live g1023 pass: 175s per
            # integral and 2-9.7GB RSS against a 2GB request, because every
            # worker built each expansion to completion before anything could
            # reject it. Capped at 20000 the same explosion aborts in 5.3s.
            for _k, _dflt in (('SAILIR_ROUTE_MAX_EXPAND', '20000'),
                              ('SAILIR_ROUTE_MAX_S', '5'),
                              ('SAILIR_ROUTE_MAX_TERMS', '200')):
                _env[_k] = os.environ.get(_k, _dflt)
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
        of = os.path.join(rdir, "out", f"rt_{tag}_{i // BATCH}.pkl")
        chunk = cand[i:i + BATCH]
        if BATCH == 1:
            # Pass the integral in argv -- no per-job file to create or clean up.
            # UNDERSCORE-SEPARATED, NOT COMMA: `queue bf,of from <file>` splits
            # each line on COMMAS, so a comma-tuple here is torn apart. Measured
            # on a 5-job smoke test, the integral 1,1,1,1,1,1,1,-2,1,1,0,0,0,0,0
            # arrived as bf="1", of="1,1,1,1,1,1,-2,1,1,0,0,0,0,0" with the real
            # output path shunted into a third argument -- every job would have
            # failed. '_' is the separator the tag convention already uses, and
            # it preserves the '-' that convention warns must not be stripped.
            bf = "_".join(str(x) for x in chunk[0])
        else:
            bf = os.path.join(rdir, "batches", f"rt_{tag}_{i // BATCH}.txt")
            with open(bf, "w") as f:
                for t in chunk:
                    f.write(",".join(str(x) for x in t) + "\n")
        batches.append((bf, of))

    listf = os.path.join(rdir, f"rt_{tag}.list")
    with open(listf, "w") as f:
        for bf, of in batches:
            f.write(f"{bf} {of}\n")
    subf = os.path.join(rdir, f"rt_{tag}.sub")
    # FILE-COUNT CONTROL. At BATCH=1 a per-process .out AND .err doubles the
    # inode cost of the whole pass for output we never read -- the batch summary
    # only matters when a job covers many integrals. Inodes are not scarce here
    # (2% used) but /het/p4 is 92% full, and every tiny file still burns a full
    # block. Keep per-process logs only when a job carries a batch worth
    # summarising; otherwise write to the shared userlog and /dev/null.
    if BATCH > 1:
        out_spec = f"{rdir}/logs/rt_{tag}_$(Process).out"
        err_spec = f"{rdir}/logs/rt_{tag}_$(Process).err"
    else:
        out_spec = "/dev/null"
        err_spec = f"{rdir}/logs/rt_{tag}_err_$(Process).err"
    sub_tmpl = f"""universe = vanilla
executable = {wsh}
arguments = "$(bf) $(of)"
request_cpus = 1
request_memory = 2GB
request_disk = 1GB
log = {rdir}/logs/rt_{tag}.log
output = {out_spec}
error = {err_spec}
queue bf,of from {listf}
"""
    # DRIP SUBMIT. One condor_submit of every job jams the schedd at this scale
    # -- PITFALL 11 records 15,000 jobs leaving it unable to answer condor_q OR
    # condor_rm, and BATCH=1 means ~104k jobs. Submit in chunks, waiting for the
    # queue to fall below QUEUE_CEILING first. The feeder checks BEFORE each
    # chunk, so the true ceiling is QUEUE_CEILING + SUBMIT_CHUNK.
    t0 = time.time()
    zero_polls = 0
    clusters = []
    nsub = 0
    for c0 in range(0, len(batches), SUBMIT_CHUNK):
        while True:
            try:
                q = subprocess.run(["condor_q", "dshih", "-af", "ClusterId"],
                                   capture_output=True, text=True, timeout=60)
                depth = len(q.stdout.split())
            except Exception:
                depth = QUEUE_CEILING + 1      # unreachable schedd => WAIT
            if depth <= QUEUE_CEILING:
                break
            print(f"[route-condor] queue at {depth} > {QUEUE_CEILING}, waiting",
                  flush=True)
            time.sleep(30)
        chunk = batches[c0:c0 + SUBMIT_CHUNK]
        cl = os.path.join(rdir, f"rt_{tag}_{c0}.list")
        with open(cl, "w") as f:
            for bf, of in chunk:
                f.write(f"{bf} {of}\n")
        with open(subf, "w") as f:
            f.write(sub_tmpl.format(listf=cl))
        r = subprocess.run(["condor_submit", subf], capture_output=True,
                           text=True, timeout=120)
        for ln in (r.stdout or "").splitlines():
            if "submitted to cluster" in ln:
                clusters.append(ln.split("cluster")[1].strip().rstrip("."))
        nsub += len(chunk)
        print(f"[route-condor] submitted {nsub}/{len(batches)} jobs "
              f"({BATCH} integral(s) each): "
              f"{r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[:100]}",
              flush=True)
    cluster = clusters[0] if clusters else None
    while time.time() - t0 < timeout_s:
        missing = [of for _, of in batches if not os.path.exists(of)]
        if not missing:
            break
        # POLL EVERY CLUSTER, not just the first. Drip submission produces one
        # cluster per chunk (21 of them for a 104k pass at 5000/chunk), and
        # polling clusters[0] alone would report an empty queue the moment that
        # first chunk drained -- abandoning every still-running chunk to the
        # serial fallback.
        alive = 1
        if clusters:
            alive = 0
            for _cl in clusters:
                try:
                    q = subprocess.run(["condor_q", _cl, "-af", "ProcId"],
                                       capture_output=True, text=True,
                                       timeout=60)
                    alive += len(q.stdout.split())
                except Exception:
                    alive += 1          # unreachable schedd => assume alive
        if alive == 0 and missing:
            # REQUIRE CONSECUTIVE ZEROES. condor_q can transiently report an
            # empty queue in the seconds after a submit, before the jobs are
            # negotiated. Believing a single reading declared 5 healthy jobs
            # dead 5s after submitting them -- and their outputs landed
            # normally moments later. Confirm before giving up.
            zero_polls += 1
            if zero_polls >= 6:
                print(f"[route-condor] {len(missing)} batch outputs missing "
                      f"with no jobs left in {zero_polls} consecutive polls "
                      f"— computing those serially", flush=True)
                break
        else:
            zero_polls = 0
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
            # bf is a PATH at BATCH>1 but a literal '_'-joined integral at
            # BATCH=1, where jobs carry the integral in argv and no batch file
            # is written at all. Opening it as a path raised FileNotFoundError
            # and took the whole routing pass down with it.
            if not os.path.exists(bf):
                t = tuple(int(x) for x in bf.split("_" if "_" in bf else ","))
                out[t] = _route(t)
                continue
            for ln in open(bf):
                ln = ln.strip()
                if ln:
                    t = tuple(int(x) for x in ln.split(","))
                    out[t] = _route(t)   # raw (uncomposed) rule — still exact
    # tidy batch inputs; keep outputs until next tag reuses names
    for bf, _ in batches:
        if BATCH == 1:
            break        # bf is an integral string, not a path
        try:
            os.remove(bf)
        except OSError:
            pass
    n_r = sum(1 for v in out.values() if v is not None)
    print(f"[route-condor] merged {len(out)} routings ({n_r} routable) in "
          f"{time.time() - t0:.0f}s", flush=True)
    return out
