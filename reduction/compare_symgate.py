#!/usr/bin/env python
"""Compare two finished orchestrator reductions of the SAME integral, symmetry
OFF vs ON (cold cache, identical config — only SAILIR_SYMMETRY differs). Reports,
cleanest metric first:

  GATE   master decomposition identical mod prime           (prerequisite)
  PRIMARY  distinct integrals reduced (tree size)           (deterministic)
  MECH     symmetry firings in ON (fired / solved-outright)  (explains PRIMARY)
  2ndary   total model beam-steps                            (count-based, clean)
  INDIC.   active wall-clock window                          (NOISY — contended)

Both runs are deterministic (--v7-cpus 1, cold cache, same model/beam/topology),
differing in exactly one bit, so every delta is attributable to symmetry.
Usage: compare_symgate.py OFF_DIR ON_DIR
"""
import pickle, sys, os, glob, re
sys.path.insert(0, "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2")
from sailir.topology import Topology
from sailir import ibp_env
from sailir.ibp_env import set_prime, set_paper_masters_only, is_master

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
off_dir = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/probetop84_off"
on_dir  = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/results/probetop84_on"

topo = Topology.from_dir(ROOT + "/topology_input/pentagonbox_nosym")
ibp_env.init_from_topology(topo); set_prime(1009); set_paper_masters_only(True)

_STEP = re.compile(r"\[v6 step\s+\d+\]\s+beam=")   # one real model beam-step

def scan(d):
    """Single pass over a run dir: completed/dispatched worker counts, summed
    model beam-steps, and (symmetry) fired / solved-outright worker counts."""
    pkls = glob.glob(f"{d}/work/results/*.pkl")          # distinct integrals reduced
    outs = glob.glob(f"{d}/work/logs/*.out")             # dispatch attempts
    steps = fired = solved = 0
    for f in outs:
        fr = so = False
        try:
            with open(f, "r", errors="ignore") as fh:
                for ln in fh:
                    if "beam=" in ln and _STEP.search(ln):       steps += 1
                    elif "symmetry reductions (total" in ln:     fr = True
                    elif "drained after symmetry" in ln:         so = True
        except OSError:
            continue
        fired  += fr
        solved += so
    span = 0.0
    if pkls:
        mt = [os.path.getmtime(p) for p in pkls]
        span = (max(mt) - min(mt)) / 3600.0
    return dict(workers=len(pkls), dispatched=len(outs),
                steps=steps, fired=fired, solved=solved, span_h=span)

def final_expr(d):
    p = f"{d}/reduction.pkl"
    if not os.path.exists(p):
        return None
    return pickle.load(open(p, "rb")).get("final_expr", {})

so, sn = scan(off_dir), scan(on_dir)
fo, fn = final_expr(off_dir), final_expr(on_dir)

print("=== OFF vs ON symmetry impact ===")
print(f"    OFF = {off_dir}")
print(f"    ON  = {on_dir}\n")

# ---- GATE ----
if fo is None or fn is None:
    miss = [n for n, f in (("OFF", fo), ("ON", fn)) if f is None]
    print(f"GATE     reduction.pkl not yet written for: {', '.join(miss)} "
          f"(run still in progress — metrics below are partial)\n")
else:
    mo = {k: v for k, v in fo.items() if is_master(k)}
    mn = {k: v for k, v in fn.items() if is_master(k)}
    if mo == mn:
        print(f"GATE     master decomposition identical mod 1009?  PASS "
              f"({len(mo)} masters, all coeffs equal)\n")
    else:
        diff = [k for k in set(mo) | set(mn) if mo.get(k) != mn.get(k)]
        print(f"GATE     master decomposition identical mod 1009?  *** FAIL *** "
              f"({len(diff)} differing, e.g. {diff[:3]})\n")

# ---- PRIMARY: tree size ----
no, nn = so["workers"], sn["workers"]
red = (1 - nn / no) * 100 if no else float("nan")
print("PRIMARY  distinct integrals reduced (deterministic, contention-free):")
print(f"    OFF workers={no:>6}   dispatched(.out)={so['dispatched']}")
print(f"    ON  workers={nn:>6}   dispatched(.out)={sn['dispatched']}")
print(f"    tree reduction = 1 - {nn}/{no} = {red:.1f}%   ({no-nn} fewer distinct integrals)\n")

# ---- MECH: symmetry firings ----
print("MECH     symmetry firings in ON:")
print(f"    workers where symmetry fired        = {sn['fired']}")
print(f"    workers SOLVED OUTRIGHT by symmetry = {sn['solved']}  (path_len=1, 0 model steps)\n")

# ---- 2ndary: total model work ----
mso, msn = so["steps"], sn["steps"]
sred = (1 - msn / mso) * 100 if mso else float("nan")
print("2ndary   total model beam-steps (count-based, contention-free):")
print(f"    OFF beam-steps={mso:>8}")
print(f"    ON  beam-steps={msn:>8}")
print(f"    model-work reduction = 1 - {msn}/{mso} = {sred:.1f}%\n")

# ---- INDIC: wall-clock (noisy) ----
print("INDICATIVE ONLY (NOISY — overlapping runs + other users' load; do not headline):")
print(f"    OFF active window ~ {so['span_h']:.2f} h   ON active window ~ {sn['span_h']:.2f} h")
