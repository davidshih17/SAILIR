#!/usr/bin/env python
"""Does acting on LEX produce longer reductions than acting on MINSUMW?

The n=1 observation that raised the question: on 1,3,3,1,1,1,1 the lex walk took
280 steps where minsumw took 128 and random 172. If that 2.2x holds across the
corpus it matters twice over -- the training data doubles, and the model is
being taught to imitate a materially slower reduction policy. But cross-policy
walk length already varies wildly per target in the data we have (3,1,3 is 273
under minsumw and 676 under random; 1,4,1 is 561 vs 280, the other direction),
so a single pair proves nothing.

Reads `samples=N` from each campaign's STATS line rather than counting jsonl
rows -- the logs are small text, the recordings are gigabytes.

Pairs strictly by target and reports the ratio distribution, so partial
campaigns are fine: only targets present in both are compared, and the count is
printed so the result can be read with the right sample size in mind.
"""
import re, glob, os, sys
import numpy as np

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/truth"
# Field ORDER on the STATS line is not guaranteed and is not the same as the
# order it reads in prose -- the real line runs
#   STATS target=... sector=... L=.. dots=.. closure=.. success=True samples=..
# so each field is matched independently rather than in one positional regex.
_TGT = re.compile(r"STATS target=([0-9,\-]+)")
_SAMP = re.compile(r"\bsamples=(\d+)")
_OK = re.compile(r"\bsuccess=(\w+)")


def scan(camp):
    out = {}
    for f in glob.glob(os.path.join(ROOT, camp, "logs", "*.out")):
        try:
            with open(f, errors="ignore") as fh:
                for ln in fh:
                    if "STATS target=" not in ln:
                        continue
                    t, s, ok = (_TGT.search(ln), _SAMP.search(ln),
                                _OK.search(ln))
                    if t and s and ok and ok.group(1) == "True":
                        out[t.group(1)] = int(s.group(1))
                    break
        except OSError:
            continue
    return out


lex = scan("lex_corpus")
mins = scan("canon_corpus")
rand = scan("rand_corpus")
print(f"recorded: lex {len(lex)}  minsumw {len(mins)}  random {len(rand)}")

both = sorted(set(lex) & set(mins))
print(f"targets in BOTH lex and minsumw: {len(both)}\n")
if not both:
    sys.exit("no overlap yet -- rerun once more lex recordings have landed")

L = np.array([lex[t] for t in both], float)
M = np.array([mins[t] for t in both], float)
# Some targets record samples=0 -- the start integral is already a master or
# the active bucket is empty, so no step is ever taken. Those carry no ratio;
# excluding them keeps the percentiles finite instead of NaN-poisoning all of
# them, and the count is printed so the exclusion is visible.
nz = M > 0
n_zero = int((~nz).sum())
r = L[nz] / M[nz]
if n_zero:
    print(f"note: {n_zero} targets have minsumw samples=0 (no step taken); "
          f"excluded from the per-target ratio, kept in the totals\n")

print(f"{'':22s} {'lex':>9} {'minsumw':>9}")
print(f"{'total steps':22s} {int(L.sum()):>9} {int(M.sum()):>9}"
      f"   ratio {L.sum() / M.sum():.3f}")
print(f"{'median steps':22s} {int(np.median(L)):>9} {int(np.median(M)):>9}")
print(f"{'mean steps':22s} {L.mean():>9.1f} {M.mean():>9.1f}")
print(f"\nper-target ratio lex/minsumw over {len(both)} paired targets:")
print(f"  median {np.median(r):.3f}   mean {r.mean():.3f}")
for q in (5, 25, 50, 75, 95):
    print(f"  p{q:<3d} {np.percentile(r, q):.3f}")
print(f"  lex LONGER on {int((r > 1).sum())}/{len(r)} targets "
      f"({(r > 1).mean():.1%}), SHORTER on {int((r < 1).sum())}, "
      f"equal on {int((r == 1).sum())}")

# The aggregate ratio is what decides corpus size and training cost; the median
# per-target ratio is what describes the typical reduction. They differ when a
# few huge targets dominate, so both are reported.
big = sorted(both, key=lambda t: -mins[t])[:10]
print("\nthe 10 longest minsumw walks, paired:")
print(f"  {'target':34s} {'minsumw':>8} {'lex':>8} {'ratio':>7}")
for t in big:
    print(f"  {t:34s} {mins[t]:>8} {lex[t]:>8} {lex[t] / mins[t]:>7.2f}")

r3 = [(t, lex[t], mins[t], rand[t]) for t in both if t in rand]
if r3:
    A = np.array([[a, b, c] for _, a, b, c in r3], float)
    print(f"\nthree-way, {len(r3)} targets present in all campaigns:")
    print(f"  total steps  lex {int(A[:, 0].sum())}  "
          f"minsumw {int(A[:, 1].sum())}  random {int(A[:, 2].sum())}")
    print(f"  vs minsumw:  lex {A[:, 0].sum() / A[:, 1].sum():.3f}x   "
          f"random {A[:, 2].sum() / A[:, 1].sum():.3f}x")
    print("  -> if lex sits near random's ratio, acting-policy choice simply "
          "costs some steps and lex is not special")
