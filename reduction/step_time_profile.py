#!/usr/bin/env python
"""Where does a worker's wall time actually go, step by step?"""
import re, sys
import numpy as np

f = sys.argv[1]
pat = re.compile(
    r'\[v6 step\s+(\d+)\].*?nm=(\d+) rs=(\d+) rs_vsz=(\d+) expr=(\d+).*?'
    r't_step=([\d.]+)s t_total=([\d.]+)s')
rows = []
for line in open(f):
    m = pat.search(line)
    if m:
        rows.append([int(m.group(1)), int(m.group(2)), int(m.group(3)),
                     int(m.group(4)), int(m.group(5)),
                     float(m.group(6)), float(m.group(7))])
a = np.array(rows, dtype=float)
step, nm, rs, vsz, expr, ts, tt = a.T
print(f"  {len(a)} steps parsed, total {tt[-1]:.0f}s")
print(f"  step 1 (startup incl. model load): {ts[0]:.1f}s")
body = ts[1:]
print(f"  remaining {len(body)} steps: sum {body.sum():.0f}s  "
      f"mean {body.mean():.2f}s  median {np.median(body):.2f}s  max {body.max():.1f}s")
print()
print("  decile profile of the walk (step range: mean t_step, mean expr, mean rs_vsz)")
n = len(a); idx = np.linspace(0, n, 11).astype(int)
for i in range(10):
    lo, hi = idx[i], idx[i+1]
    if hi <= lo: continue
    sl = slice(lo, hi)
    print(f"    steps {int(step[lo]):4d}-{int(step[hi-1]):4d}:  "
          f"t_step {ts[sl].mean():6.2f}s   expr {expr[sl].mean():8.0f}   "
          f"rs_vsz {vsz[sl].mean():9.0f}   nm {nm[sl].mean():7.0f}")
print()
cum = np.cumsum(ts)
for frac in (0.5, 0.8, 0.9):
    k = int(np.searchsorted(cum, cum[-1]*frac))
    print(f"  {int(frac*100)}% of the wall time is spent by step {int(step[k])} "
          f"({100.0*(k+1)/n:.0f}% of steps)")
print()
print("  slowest 10 steps:")
for i in np.argsort(ts)[-10:][::-1]:
    print(f"    step {int(step[i]):4d}: t_step={ts[i]:6.1f}s  expr={int(expr[i]):6d}  "
          f"rs_vsz={int(vsz[i]):7d}  nm={int(nm[i]):5d}")
