#!/usr/bin/env python
"""SYMMETRY-ROUTING CLOSURE WORKER (Condor batch, 2026-07-30).

Input: a text file of integrals (one comma-tuple per line, ~100 per batch).
For each, compute the COMPOSED routing rule: route the integral, substitute
routable terms recursively (memoized) until only non-routable survivors
remain — one flat rule  I = sum c_J * J  over survivors. Routing rules are
strictly descending, so the recursion is a finite DAG walk (no cycles).

Output pkl: {integral: composed_rule_dict or None (= not routable)}.
The orchestrator stores these in its cache exactly like worker results; the
central expression then only ever sees survivors — the intermediate routed
integrals never touch the orchestrator's fold, cache growth, or GC.
"""
import os
import sys
import pickle

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reduction"))

from symmetry_route import canonical_monolithic_rule as _route

# COMPOSITION PRIME, taken from the module that actually SOLVES the routes.
# symmetry_route uses topo_config.canonicalize_module(), which for gravity3L is
# canonicalize2 -- NOT canonicalize_GR -- and canonicalize2 derives P from its
# store's prod_point, so it is self-consistent by construction. Binding P to
# canonicalize_GR here was wrong twice over: that module is not on this code
# path, and with SAILIR_SYM_PRIME set it would compose mod-101 while the routes
# were solved mod-1009. A hardcoded 1009 was also wrong -- it silently assumed
# the store's prime. Take it from the solver.
import topo_config as _tc
P = _tc.canonicalize_module().P
infile, outfile = sys.argv[1], sys.argv[2]

# ACCEPT A LITERAL INTEGRAL, not just a batch file. At one integral per Condor
# job a batch file is a one-line file per job -- 104k of them for a single
# routing pass, all inodes and block slack for data that fits in argv. If argv[1]
# parses as a comma-tuple and is not an existing path, take it directly.
integrals = []
if not os.path.exists(infile) and ("," in infile or "_" in infile):
    # UNDERSCORE is what routing_condor sends at BATCH=1: Condor's
    # `queue bf,of from <file>` splits on commas, so a comma-tuple in argv is
    # split across fields. Accept either so a hand-run with commas still works.
    sep = "_" if "_" in infile else ","
    integrals.append(tuple(int(x) for x in infile.split(sep)))
else:
    with open(infile) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                integrals.append(tuple(int(x) for x in ln.split(",")))

raw = {}
# preload the orchestrator's persisted routing table (raw or composed rules —
# both are exact identities): known routes become dictionary lookups instead
# of fresh solves. outfile = <work>/routing/out/<batch>.pkl -> <work>/sym_memo.pkl
_smp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(outfile)))), "sym_memo.pkl")
_preloaded = 0
if os.path.exists(_smp):
    try:
        with open(_smp, "rb") as _f:
            raw.update(pickle.load(_f))
        _preloaded = len(raw)
        print(f"preloaded {_preloaded} persisted routing rules", flush=True)
    except Exception as _e:
        print(f"could not preload {_smp}: {_e}", flush=True)


# SIZE CAP. Routing a high-numerator integral can EXPLODE: measured on the real
# frontier, s=20 produced a 1,315,600-term rule (against a 714,538-term
# expression) with only 8 masters in it. Such a rule is expression GROWTH, not
# reduction -- an IBP reduction of the same integral emits a median of 6-8
# terms, so dispatching a worker is strictly better.
#
# The cap is on the OUTPUT size, not on s: the blowup is bimodal and NOT
# monotonic in s. In the same sample s=24 routed to 1 term while s=20 gave
# 1.3M, and the >=20 bucket was {1, 15, 28, 1315600}. An s-threshold would
# reject the cheap wins and still admit the disasters.
#
# Over-cap integrals are treated exactly like non-routable ones (None), so they
# fall through to the normal worker dispatch.
_MAX_TERMS = int(os.environ.get('SAILIR_ROUTE_MAX_TERMS', '200'))
# NUMERATOR-DEGREE GATE. Skip routing entirely for integrals above this total
# numerator degree; they go to worker dispatch instead. Checked BEFORE _route,
# so a skipped integral costs nothing at all -- unlike the output caps, which
# only act after the expansion has been built.
#
# s is a PROXY, not the cause. Cost comes from the cascade (canonical_monolithic
# _rule composes across transforms and recursively routes what it produces), and
# neither s nor the individual powers predict it: measured, s=24 with powers
# {3,7,14} gave ONE term while s=16 with powers {8,1,7} gave 6,435, and an s=20
# case gave 1,315,600. What s buys is PREDICTABILITY -- everything sampled at
# s<=5 produced small rules (0, 1, 16, 45 terms). The price is forgoing cheap
# high-s wins like that s=24 single-term route.
_MAX_S = int(os.environ.get('SAILIR_ROUTE_MAX_S', '5'))
n_capped = 0
n_skipped_s = 0


# PER-INTEGRAL TIME LIMIT. Routing cost is bimodal to an extreme degree:
# measured over 104,383 integrals, the MEDIAN routes in 1.2s and p90 in 3.2min,
# while a few hundred ran for HOURS -- one straggler was still going at 6.7h.
# That tail made a pass that should be free cost ~5,700 CPU-hours, defeating the
# point of routing, which exists to AVOID dispatching an IBP worker.
#
# A timeout is nondeterministic in principle (the same integral may route on a
# fast node and not on a loaded one), and that is a real cost. It is accepted
# here because the alternative measured worse: the answer is never WRONG, only
# less optimised -- a timed-out integral becomes None, i.e. a survivor, and gets
# reduced by IBP exactly as any non-routable integral does. Routing is an
# optimisation, not a requirement.
_TIME_LIMIT = int(os.environ.get('SAILIR_ROUTE_TIME_LIMIT', '300'))
n_timeout = 0


class _RouteTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _RouteTimeout()


if _TIME_LIMIT > 0:
    import signal
    signal.signal(signal.SIGALRM, _on_alarm)


def route(i):
    global n_capped, n_skipped_s, n_timeout
    if i not in raw:
        if _MAX_S >= 0 and -sum(x for x in i if x < 0) > _MAX_S:
            n_skipped_s += 1
            raw[i] = None
            return None
        if _TIME_LIMIT > 0:
            signal.alarm(_TIME_LIMIT)
        try:
            r = _route(i)
        except _RouteTimeout:
            n_timeout += 1
            r = None
        finally:
            if _TIME_LIMIT > 0:
                signal.alarm(0)
        if r is not None and len(r) > _MAX_TERMS:
            n_capped += 1
            r = None
        raw[i] = r
    return raw[i]


comp = {}       # integral -> composed rule (routable) — memoized across batch


def composed(top):
    """Iterative post-order composition over the (strictly descending) DAG."""
    stack = [top]
    while stack:
        j = stack[-1]
        if j in comp:
            stack.pop()
            continue
        r = route(j)
        if r is None:
            raise AssertionError("composed() called on non-routable integral")
        deps = [k for k in r if k not in comp and route(k) is not None]
        if deps:
            stack.extend(deps)
            continue
        out = {}
        for k, c in r.items():
            sub = comp[k] if route(k) is not None else None
            if sub is None:
                out[k] = (out.get(k, 0) + c) % P
            else:
                for l, cl in sub.items():
                    v = (out.get(l, 0) + c * cl) % P
                    if v:
                        out[l] = v
                    else:
                        out.pop(l, None)
        comp[j] = {k: v for k, v in out.items() if v}
        stack.pop()
    return comp[top]


import time as _time
_t0 = _time.time()
res = {}
n_routable = 0
for _n, i in enumerate(integrals):
    if route(i) is None:
        res[i] = None
    else:
        res[i] = composed(i)
        n_routable += 1
    if (_n + 1) % 10 == 0:
        print(f"  ... {_n + 1}/{len(integrals)} done ({n_routable} routable), "
              f"{len(raw) - _preloaded} fresh route solves, "
              f"{_time.time() - _t0:.0f}s", flush=True)

with open(outfile + ".tmp", "wb") as f:
    pickle.dump(res, f)
os.replace(outfile + ".tmp", outfile)
sizes = sorted(len(v) for v in res.values() if v is not None)
print(f"  timed out {n_timeout} integral(s) after {_TIME_LIMIT}s; "
      f"skipped {n_skipped_s} integral(s) with s>{_MAX_S}; "
      f"capped {n_capped} route(s) over {_MAX_TERMS} terms "
      f"(both -> worker dispatch)", flush=True)
print(f"batch {os.path.basename(infile)}: {len(integrals)} integrals, "
      f"{n_routable} routable; composed-rule terms "
      f"median={sizes[len(sizes)//2] if sizes else 0} "
      f"max={sizes[-1] if sizes else 0}", flush=True)
