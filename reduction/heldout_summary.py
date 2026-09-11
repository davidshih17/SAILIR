#!/usr/bin/env python
"""Aggregate the held-out model comparison.

Five candidate checkpoints scored on the SAME 8 held-out lex walks, so every
difference is the model and nothing else. Reads the per-(model, walk) pickles
rank_on_recording.py wrote.

PRIMARY METRIC: LEX-in-top-20, POOLED over every scored step. Pooled rather
than mean-of-walks so each state counts once -- a 27-step walk should not carry
the same weight as a 563-step one. The per-walk spread is printed alongside
because averaging hid an n=1 trap earlier in this work.

The threshold metric alone can conceal real differences (a model can sit just
inside or just outside 20 on many states), so median/p90 rank and the outright
top-1 rate are reported too.

ANY-correct-in-top-20 is what the beam strictly requires -- some valid action
inside its K=20 expansion -- but it sits near ceiling (0.91-0.98 measured), so
it discriminates poorly and is reported last.
"""
import os, glob, pickle
import numpy as np

D = ("/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/truth/"
     "lex_corpus/heldout/out")
LABEL = {
    'ftlex':        'ft_lex          (incumbent, off-policy corpus)',
    'onpol_ep15':   'ft_onpolicy ep15 (best val_loss  1.2831)',
    'onpol_ep41':   'ft_onpolicy ep41 (best val_top20 0.9606)',
    'onpol_ep57':   'ft_onpolicy ep57 (best val_top1  0.7414)',
    'scratch_ep23': 'ft_scratch  ep23 (best val_top20 0.9671, random init)',
}
ORDER = ['ftlex', 'onpol_ep15', 'onpol_ep41', 'onpol_ep57', 'scratch_ep23']

data = {}
for tag in ORDER:
    rows = []
    for p in sorted(glob.glob(os.path.join(D, tag, '*.pkl'))):
        with open(p, 'rb') as f:
            d = pickle.load(f)
        rows.append((os.path.basename(p), d['rows']))
    if rows:
        data[tag] = rows

if not data:
    raise SystemExit('no results yet')

nwalk = {t: len(v) for t, v in data.items()}
print(f"walks scored per model: {nwalk}")
done = [t for t in ORDER if nwalk.get(t, 0) == 8]
print(f"complete (8/8): {done or 'none yet'}\n")

print(f"{'model':52s} {'steps':>6} {'TOP20':>7} {'top1':>7} {'med':>5} "
      f"{'p90':>6} {'max':>6} {'ANY20':>7}")
print('-' * 100)
summary = {}
for tag in ORDER:
    if tag not in data:
        continue
    L, A = [], []
    per_walk = []
    for _, rows in data[tag]:
        l = [r['lexrank'] for r in rows]
        L += l
        A += [r['anyrank'] for r in rows if r.get('anyrank') is not None]
        if l:
            per_walk.append(float(np.mean(np.array(l) <= 20)))
    L = np.array(L, float)
    A = np.array(A, float) if A else None
    top20 = float(np.mean(L <= 20))
    summary[tag] = (top20, per_walk)
    print(f"{LABEL[tag]:52s} {len(L):>6} {top20:>7.4f} "
          f"{np.mean(L == 1):>7.4f} {int(np.median(L)):>5} "
          f"{np.percentile(L, 90):>6.0f} {int(L.max()):>6} "
          f"{(np.mean(A <= 20) if A is not None else float('nan')):>7.4f}")

print("\nper-walk LEX-in-top-20 (spread matters -- a single walk misled this "
      "analysis once):")
for tag in ORDER:
    if tag in summary:
        pw = summary[tag][1]
        print(f"  {tag:14s} " + ' '.join(f"{x:.3f}" for x in sorted(pw)) +
              f"   min {min(pw):.3f}  max {max(pw):.3f}")

if len(summary) > 1:
    best = max(summary, key=lambda t: summary[t][0])
    base = summary.get('ftlex', (None,))[0]
    print(f"\nBEST pooled LEX-in-top-20: {best} at {summary[best][0]:.4f}")
    if base is not None and best != 'ftlex':
        print(f"  vs incumbent ft_lex {base:.4f}  ->  "
              f"{summary[best][0] - base:+.4f}")
