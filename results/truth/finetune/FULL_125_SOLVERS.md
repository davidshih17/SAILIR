# Configurations that have solved all 125 test integrals

As of 2026-09-03 there are **two**, and no others. Both reach 100/100 on
test100 and 25/25 on test25 with zero `STUCK` and `nm=0` on every target (every
active bucket fully drained — no partial credit anywhere).

| | config | date | corpus | prime |
|---|---|---|---|---|
| 1 | **K=1000 ftcorrupt** `nosubs` ep19, val_loss 0.5400 | 2026-09-01 | `ftcull1848_corrupt_shards`, 1843/1848 targets, 203,728 rows | 1009 |
| 2 | **eqact T=10** prob40 | 2026-09-03 | same corpus line, equation branch T=10 | 1009 |

Config 1 is the subject of `MILESTONE_FULL_125_SOLVE.md` (model, corpus and
beam settings in full). Config 2 differs by using the equation model at T=10,
which was selected because it beat config 1 on val_loss (0.5242 @ep8 vs
0.5400 @ep19).

---

## Head-to-head, prob40 vs prob40, identical targets

All numbers counted from the terminal marker `[v7-worker] SUCCESS in` in the
worker's own log, using the seconds the worker reported. NOT `.pkl` counts
(retries leave extra files: 107 pkls for 100 targets, 45 for 25) and NOT file
ctime (inode change times drift). Reproduce with `compare_eqact_vs_k1000.py`.

### test100 (100 targets)

| arm | solved | median s | p90 s | max s | total CPU-h | max h | med len | max len | max RSS MB |
|---|---|---|---|---|---|---|---|---|---|
| eqact T=10 prob40 | 100/100 | 85 | 1,964 | 73,029 | **49.3** | 20.3 | 8 | 886 | 6,894 |
| K=1000 prob40 | 100/100 | 11 | 278 | 35,394 | **25.2** | 9.8 | 6 | 1309 | 6,513 |
| K=1000 prob80 | 100/100 | 16 | 563 | 33,769 | 33.3 | 9.4 | 6 | 988 | 4,761 |

### test25 (25 targets — the hard set)

| arm | solved | median s | p90 s | max s | total CPU-h | max h | med len | max len | max RSS MB |
|---|---|---|---|---|---|---|---|---|---|
| eqact T=10 prob40 | 25/25 | 11,397 | 51,814 | **71,033** | 149.2 | **19.7** | 326 | 1429 | **7,399** |
| K=1000 prob40 | 25/25 | 6,333 | 55,859 | 117,522 | **135.5** | 32.6 | 487 | 2408 | 10,583 |
| K=1000 prob80 | 25/25 | 7,416 | 56,597 | 120,001 | 145.6 | 33.3 | 459 | 2018 | 7,290 |

---

## The verdict: eqact is a wash, slightly worse

* test100 — **1.96x slower** (49.3 vs 25.2 CPU-h), faster on only 11/100.
* test25 — **1.10x slower** (149.2 vs 135.5 CPU-h), faster on 8/25.

### Why: decision quality vs per-step cost (paired per target)

| set | STEPS eqact/K1000 | SEC/STEP eqact/K1000 |
|---|---|---|
| test100 | median 1.00x, fewer on 27% | **5.30x** |
| test25 | median **0.72x**, fewer on 72% | **3.25x** |

Ratios are formed per target then summarised — a ratio of medians is not a
median of ratios, and on a set this long-tailed the two disagree badly.

On **test100 eqact is pure overhead**: identical step counts, 5.3x the compute
per step. The equation branch buys nothing on easy targets.

On **test25 the equation branch genuinely works** — 28% fewer steps, fewer on
72% of targets, median path 326 vs 487, max path 1429 vs 2408. That is real
decision quality. It is simply not worth 3.25x per step, so the net is still
slower.

### Where eqact does win (hard set only)

* **worst case 1.65x better**: 19.7 h vs 32.6 h max.
* **peak memory lower**: 7,399 vs 10,583 MB — the only arm that stayed inside
  the 10 GB/CPU budget.

If the binding constraint is the tail (the one target that runs a day and a
half) eqact is the better pick. For throughput it is not.

### A note on checkpoint selection

eqact T=10 was chosen because it beat the baseline on val_loss. It did, and it
still lost on wall-clock. Another data point for the standing rule that
val_loss alone is the wrong selection metric — beam solve behaviour has to be
measured directly. See `sailir_checkpoint_selection.md`.

---

## Caveats on these numbers

* The K=1000 **test25 prob40** results are pooled from `beamexp/` and
  `beamexp25/` because of the EXPDIR split-path bug (see the milestone doc): 12
  targets landed in one tree, 13 in the other. Deduped by target, all 25 present
  and each counted once — but they ran under two submissions, not one.
* eqact was run at **prob40 only**. There is no eqact prob80 arm, so the width
  comparison exists for config 1 only (where width was not the discriminator).
* Both configs are p=1009. Neither has been re-run at p=101.
