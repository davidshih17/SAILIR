#!/usr/bin/env python
"""Is routing LEAKING? Replay one real batch, one integral at a time, printing
per-integral wall time and RSS plus the size of every cache that could grow.

If cost is intrinsic (some integrals just hard), time per integral is noisy but
FLAT and RSS plateaus. If something leaks, both climb monotonically with the
number processed, independent of which integral is being done.
"""
import os, sys, time, resource, gc
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "reduction"))

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

import symmetry_route as SR
import topo_config as _tc
_cz = _tc.canonicalize_module()
from symmetry_route import canonical_monolithic_rule as _route
print(f"solver={_cz.__name__} P={_cz.P}", flush=True)

batch = sys.argv[1]
integrals = []
for ln in open(batch):
    ln = ln.strip()
    if ln:
        integrals.append(tuple(int(x) for x in ln.split(",")))
print(f"batch {os.path.basename(batch)}: {len(integrals)} integrals", flush=True)

def cache_sizes():
    """Every module-level dict/set in symmetry_route + the canonicalize module
    that could accumulate across calls."""
    out = []
    for mod in (SR, _cz):
        for name in dir(mod):
            if name.startswith('__'):
                continue
            v = getattr(mod, name, None)
            if isinstance(v, (dict, set, list)) and len(v) > 0:
                out.append((f"{mod.__name__}.{name}", len(v)))
    return out

base = dict(cache_sizes())
print(f"baseline caches: {sorted(base.items(), key=lambda kv:-kv[1])[:8]}", flush=True)

t_start = time.time()
for n, I in enumerate(integrals, 1):
    s = -sum(x for x in I if x < 0)
    t0 = time.time()
    try:
        r = _route(I)
        nterms = 0 if r is None else len(r)
    except Exception as e:
        nterms = -1
        print(f"  ERR {type(e).__name__}", flush=True)
    dt = time.time() - t0
    if n <= 40 or n % 5 == 0:
        cur = dict(cache_sizes())
        grown = sorted(((k, v - base.get(k, 0)) for k, v in cur.items()
                        if v - base.get(k, 0) > 0), key=lambda kv: -kv[1])[:4]
        print(f"  {n:3d}/{len(integrals)} s={s} terms={nterms:6d} "
              f"{dt:8.2f}s  rss={rss_mb():8.1f}MB  total={time.time()-t_start:7.0f}s  "
              f"grown={grown}", flush=True)
print(f"DONE {time.time()-t_start:.0f}s rss={rss_mb():.1f}MB", flush=True)
