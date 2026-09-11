#!/usr/bin/env python
"""Where does a beam run go off the rails, and does it ever recover?

The beam logs one line per step carrying `nm=` -- the number of ACTIVE
non-masters left in the best state. That is the progress measure the search is
actually trying to drive to zero, so its trajectory is the honest record of
what the model did: monotone descent means steady progress, a ratchet upward
means the beam took a substitution that introduced more work than it removed,
and a long flat stretch means it is churning without draining.

WHY THIS RATHER THAN RANK METRICS. Every rank metric tried so far -- val_top1,
val_top20, val_loss, and LEX-in-top-20 on held-out walks -- ranked these
checkpoints in an order that did NOT match their beam step counts. nm is not a
proxy: it is the quantity the beam terminates on.

Reported per run:
  peak / final nm, and the step where nm first exceeded its starting value
  n_ratchets  -- how many times nm rose from one logged step to the next
  worst_rise  -- the largest single-step increase (the biggest single mistake)
  recovered   -- whether nm came back below the pre-rise level afterwards, and
                 how many steps that took. This is the number that decides
                 whether "let the beam recover better" is even the right fix:
                 if runs never recover, the damage is permanent and the answer
                 is to avoid the bad move; if they do recover but slowly, the
                 answer is to widen or diversify the beam.
"""
import os, re, sys, glob
import numpy as np

H = ("/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/truth/"
     "finetune/hard4_ckpt/logs")
TRUTH = {'1_2_1_1_1_1_1_1_0_1_0_0_0_0_0': 348,
         '2_1_0_1_1_1_1_1_1_1_0_0_0_0_0': 307,
         '2_1_1_2_1_1_1_0_0_0_0_0_0_0_0': 528,
         '1_1_0_2_2_0_1_0_1_1_0_0_0_0_0': 525}
MODELS = ['onpol_ep15', 'scratch_ep23', 'onpol_ep41', 'onpol_ep57']
STEP = re.compile(r"^\[v6 step (\d+)\] beam=.*?nm=(\d+).*?expr=(\d+)")


def load(path):
    st, nm, ex = [], [], []
    try:
        with open(path, errors='ignore') as f:
            for ln in f:
                m = STEP.match(ln)
                if m:
                    st.append(int(m.group(1)))
                    nm.append(int(m.group(2)))
                    ex.append(int(m.group(3)))
    except OSError:
        pass
    return np.array(st), np.array(nm), np.array(ex)


print(f"{'model':14s} {'target':13s} {'steps':>6} {'truth':>6} {'nm0':>4} "
      f"{'peak':>5} {'final':>5} {'ratch':>6} {'worst':>6} {'%up':>5}")
print('-' * 88)
rows = {}
for m in MODELS:
    for t, tr in TRUTH.items():
        st, nm, ex = load(os.path.join(H, f"{m}__{t}.out"))
        if len(nm) < 10:
            continue
        d = np.diff(nm)
        rises = d[d > 0]
        solved = os.path.exists(os.path.join(
            os.path.dirname(H), 'out', f"{m}__{t}.pkl"))
        rows[(m, t)] = (st, nm, ex, solved)
        print(f"{m:14s} {t[:12]:13s} {st[-1]:>6} {tr:>6} {nm[0]:>4} "
              f"{nm.max():>5} {nm[-1]:>5} {len(rises):>6} "
              f"{(rises.max() if len(rises) else 0):>6} "
              f"{100*len(rises)/max(len(d),1):>5.1f}")

# ---- recovery analysis: after the worst single rise, does nm come back? -----
print("\nRECOVERY after the largest single-step rise "
      "(does the beam undo its worst mistake?)")
print(f"{'model':14s} {'target':13s} {'at step':>8} {'rise':>6} "
      f"{'level':>6} {'recovered':>10} {'steps to':>9}")
print('-' * 76)
for (m, t), (st, nm, ex, solved) in rows.items():
    d = np.diff(nm)
    if not len(d) or d.max() <= 0:
        continue
    i = int(np.argmax(d))
    before, after = int(nm[i]), int(nm[i + 1])
    back = np.where(nm[i + 1:] <= before)[0]
    if len(back):
        rec, k = 'yes', int(back[0])
    else:
        rec, k = 'NO', -1
    print(f"{m:14s} {t[:12]:13s} {int(st[i]):>8} "
          f"{after-before:>6} {before:>6} {rec:>10} "
          f"{(k if k >= 0 else '-'):>9}")

# ---- where the run stops making progress -----------------------------------
print("\nSTALL: longest stretch with no NEW minimum in nm "
      "(the beam churning without draining)")
print(f"{'model':14s} {'target':13s} {'longest stall':>14} "
      f"{'from step':>10} {'nm there':>9} {'solved':>7}")
print('-' * 76)
for (m, t), (st, nm, ex, solved) in rows.items():
    best = nm[0]
    run = start = 0
    worst_run = worst_at = worst_nm = 0
    for j in range(len(nm)):
        if nm[j] < best:
            best = nm[j]
            if run > worst_run:
                worst_run, worst_at, worst_nm = run, int(st[start]), int(nm[start])
            run, start = 0, j
        else:
            run += 1
    if run > worst_run:
        worst_run, worst_at, worst_nm = run, int(st[start]), int(nm[start])
    print(f"{m:14s} {t[:12]:13s} {worst_run:>14} {worst_at:>10} "
          f"{worst_nm:>9} {('yes' if solved else 'NO'):>7}")
