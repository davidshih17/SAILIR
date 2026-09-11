#!/usr/bin/env python
"""Does the truth engine's CLOSURE SIZE predict what the beam search costs?

This is the one candidate predictor of difficulty that has been recorded all
along and never tested. Every hypothesis tried so far has been refuted:
distance to the training data (twice), propagator count, sector coverage. The
closure size -- how many actions the dependency closure of the target's rule
contains -- is the only remaining feature that is a property of the REDUCTION
rather than of the index vector, which is where the evidence has been pointing.

Sources, both already on disk:
  closure : results/truth/{rand,canon}_corpus/logs/*.out, the line
            "[plan] closure=N actions" (or worker_replay's "N actions")
            paired with the target printed in the same log
  cost    : the beam campaigns' worker logs, "SUCCESS in Ts path_len=N"

Prints Spearman (rank) correlation, which is the right statistic here: the
question is monotonicity, not linearity, and both variables have long tails.
"""
import os, re, glob, sys
import numpy as np

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
FT = os.path.join(ROOT, "results/truth/finetune")

# The corpus logs put everything on ONE line, unbracketed and without the word
# "actions":
#   STATS target=2,0,1,0,1,0,0,1,1,0,0,0,0,0,0 sector=405 L=5 dots=1 closure=1
#     success=True samples=1 reason=... truth_time=0.0s ...
# so one regex over that line gets target, closure and the success flag at once.
STATS = re.compile(r"STATS target=([0-9,\-]+).*?closure=(\d+).*?success=(\w+)")
SUC = re.compile(r"SUCCESS in ([0-9.]+)s path_len=(\d+)")


def norm(s):
    return tuple(int(x) for x in s.strip().split(",") if x != "")


def scan_corpus(d):
    """target -> closure size, from a recording campaign's worker logs."""
    out = {}
    for f in glob.glob(os.path.join(d, "logs", "*.out")):
        try:
            with open(f, errors="ignore") as fh:
                for ln in fh:
                    if "STATS target=" not in ln:
                        continue
                    m = STATS.search(ln)
                    if m and m.group(3) == "True":
                        out.setdefault(norm(m.group(1)), int(m.group(2)))
                    break
        except OSError:
            continue
    return out


def scan_campaign(d):
    """target -> (path_len, seconds), from a beam campaign's worker logs."""
    out = {}
    for f in glob.glob(os.path.join(d, "logs", "*.out")):
        tag = os.path.basename(f)[:-4]
        try:
            tgt = tuple(int(x) for x in tag.split("_"))
        except ValueError:
            continue
        try:
            txt = open(f, errors="ignore").read()
        except OSError:
            continue
        m = None
        for m in SUC.finditer(txt):
            pass
        if m:
            out[tgt] = (int(m.group(2)), float(m.group(1)))
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


clos = {}
for c in ("rand_corpus", "canon_corpus"):
    clos.update(scan_corpus(os.path.join(ROOT, "results/truth", c)))
print(f"closure sizes recovered for {len(clos)} targets "
      f"(min {min(clos.values())}, median {int(np.median(list(clos.values())))}, "
      f"max {max(clos.values())})\n")

for camp in ("dual100_lex_w20", "dual100_lex", "wprob100_lex", "dual6_frontier"):
    d = os.path.join(FT, camp)
    if not os.path.isdir(d):
        continue
    cost = scan_campaign(d)
    both = [(clos[t], cost[t][0], cost[t][1], t) for t in cost if t in clos]
    print(f"== {camp}: {len(cost)} solved, {len(both)} also have a closure size")
    if len(both) < 8:
        for c, s, h, t in sorted(both):
            print(f"     closure {c:5d}  steps {s:5d}  {h/3600:5.2f}h  {list(t)}")
        print()
        continue
    C = np.array([x[0] for x in both], float)
    S = np.array([x[1] for x in both], float)
    H = np.array([x[2] for x in both], float)
    print(f"   spearman(closure, steps) = {spearman(C, S):+.3f}")
    print(f"   spearman(closure, hours) = {spearman(C, H):+.3f}")
    print(f"   steps/closure ratio: median {np.median(S / C):.2f} "
          f"min {np.min(S / C):.2f} max {np.max(S / C):.2f}")
    order = np.argsort(-S)
    print("   the 5 costliest:")
    for i in order[:5]:
        print(f"     closure {int(C[i]):5d}  steps {int(S[i]):5d}  "
              f"{H[i]/3600:5.2f}h  {list(both[i][3])}")
    print()

# the three still-unsolved, whose closures the diagnostic just measured
print("== the three UNSOLVED (closure from the rank diagnostic, steps = where "
      "the beam stood, not a solve)")
for t, c, s in ((( 2,1,1,1,1,1,1,0,0,0,0,0,0,0,0), 910, 3485),
                (( 3,1,1,1,1,1,1,0,0,0,0,0,0,0,0), 787, 3339),
                (( 4,1,1,1,1,1,1,0,0,0,0,0,0,0,0), 1692, 2732)):
    print(f"     closure {c:5d}  beam at {s:5d} steps (>= {s/c:.2f}x closure)  {list(t)}")
