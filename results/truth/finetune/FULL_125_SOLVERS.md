# Configurations that have solved all 125 test integrals

FULL SESSION REPORT: `results/truth/GREEDY_FINDINGS.md` — why the beam failed
(five distinct cross-state failure modes on one integral), why log-prob sorting
cannot be rescued at any width, the sector-coverage gap, the nm-penalty
measurement error, and the g1023 run.

As of 2026-09-07 there are **three**. The third (greedy, width 1) is 65x
cheaper than either of the first two and is described immediately below;
the 2026-09-03 head-to-head between configs 1 and 2 follows unchanged.

TWO CAVEATS THAT APPLY TO EVERY NUMBER IN THIS FILE, found 2026-09-07:

1. **This 125-target set is 47% contaminated.** 59 of the 125 have truth walks
   in `p101_cull_corpus/train` (13) and `p101_cull_corpus_v2/train` (46) — the
   two runs `combine_and_pack.sh` packed into `p101_shards`. So "125/125" is
   not a generalisation claim for any p=101 model. The clean subset is the
   other 66. (Configs 1-2 are p=1009 models trained on
   `ftcull1848_corrupt_shards`, so the overlap must be re-checked per corpus
   before reusing their numbers as held-out.)
2. The genuinely held-out set is the **hard targets** — 21,761 integrals that
   blew the memory/time caps during CLOSURE generation, so they never got a
   truth walk and cannot be in any corpus (verified: 0 of 21,761 appear in
   either p101 corpus). List: `heldout_hard_clean.txt`.

---

## 3. GREEDY, width 1 (2026-09-07) — 125/125 in 2.5 CPU-hours

**No beam at all.** `SAILIR_BEAM_SORT=prob --beam_width 1`,
`SAILIR_SCORE=local`, `SAILIR_BEAM_TOTAL=1`, `--no-tabu`. Model
`checkpoints/gravity3L_p101_scratch/best_model.pt` (epoch 34), prime **101**.
Submit file `greedy125.sub`, cluster 1920017.

**NM_PEN = 0.1, NOT 0 — this run was NOT the plain argmax.** `greedy125.sub`
sets `SAILIR_NM_PENALTY=0`, but `p101_beam_worker.sh:32` carried an
UNCONDITIONAL `export SAILIR_NM_PENALTY=0.1` that overrode it; the worker's own
banner says `NM_PEN=0.1`. So the selection rule that produced 125/125 is

    (action log-prob) - 0.1 * (number of non-masters)

i.e. greedy with a mild bias toward moves that shrink the leftover pile. The
worker line is now `${SAILIR_NM_PENALTY:-0.1}` (default unchanged). ALWAYS read
the worker banner, not the submit file — see GREEDY_FINDINGS.md section 7 for
the A/B against a genuine NM_PEN=0 arm.

| set | solved | median s | p90 s | max s | total CPU-h | max h | med len | max len | max RSS MB |
|---|---|---|---|---|---|---|---|---|---|
| test100 | 100/100 | 2 | 25 | 864 | **0.64** | 0.24 | 18 | 1270 | 1,797 |
| test25 | 25/25 | 182 | 618 | 1,073 | **1.85** | 0.30 | 692 | 1751 | 1,804 |

`nm=0` on all 125, zero `INCOMPLETE`, zero cap hits, zero crashes.

### vs the previous best (K=1000 prob40)

| | greedy | K=1000 prob40 | factor |
|---|---|---|---|
| test100 CPU-h | 0.64 | 25.2 | **39x cheaper** |
| test25 CPU-h | 1.85 | 135.5 | **73x cheaper** |
| total CPU-h | **2.49** | 160.7 | **65x cheaper** |
| worst single target | 0.30 h | 32.6 h | **109x** |
| peak RSS | 1.8 GB | 10.6 GB | **5.9x lower** |

By subset: **59/59 contaminated, 66/66 clean held-out.** The clean rate is
100%, so the result does not depend on the contamination.

### Why width 1 beats width 40

Every failure diagnosed on `1_0_1_0_0_0_5_1_1_1_0_0_0_0_0` (which defeated a
40-wide beam for 34.5 h and fell to greedy in 10.6 s) was a CROSS-STATE
comparison discarding a good state: the tabu ban, the expression-only dedup,
total-weight displacement, a max-lex-weight rank of 41/41, and 1.5e17 paths
outscoring truth on summed log-prob. At width 1 no cross-state comparison ever
happens — the only judgement is "which action from THIS state", the one regime
where the model's probabilities are commensurable (each state's actions are a
separate softmax; scores from different states are not comparable, and any
tiebreak built from them fails exactly where the model is unreliable).

Measured support: along the greedy trajectory the model's confidence after an
unconfident step is statistically identical to after a confident one (median
0.9551 vs 0.9549), i.e. uncertainty marks a fork with several viable moves, not
a wrong turn. Greedy left the labelled truth path at step 3 (p=0.9477) and
solved anyway in 94 steps vs the truth walk's 85 — "off-label" was a different
valid route, not an error.

CAPS (greedy has no width bounding it): `SAILIR_MAX_STEPS`,
`SAILIR_BEAM_WALL`, and `periodic_remove = (MemoryUsage > 20480)`. At 3000
steps / 1800 s / 20 GB none of the 125 hit any cap.

### Held-out check: 300/300 (2026-09-07, cluster 1920018)

300 sampled from `heldout_hard_clean.txt` (seed 20260907), caps 10,000 steps /
24 h / 20 GB. **300/300 solved**, `nm=0` on all, zero cap hits, zero crashes;
total **0.21 CPU-h**, median 1.1 s, max 138 s, peak RSS 872 MB.

READ THIS BEFORE QUOTING 300/300. The set is genuinely never-seen, but it is
NOT a hard-REDUCTION benchmark: median path length is **1**, and 52% of the
targets (156/300) drain in a single move; only 9% need 10+ steps. Compare the
125 set, median path 27.

The reason is that "hard target" here means hard for CLOSURE GENERATION, which
is a different computation. `batch_closure.py` builds ONE SHARED LINEAR SYSTEM
per sector GROUP, and the watchdog fired while constructing it — per
`save_hard_targets.py`, "before a single target was solved" (549 group attempts
breached 9 GB, 265 breached 19 GB). When a group died, EVERY target it carried
was marked hard regardless of its own difficulty; dead groups averaged 15.4
targets against an overall mean of 7.8, i.e. the big groups died. So a
one-move integral inherits the label from its group.

Two consequences worth more than the 300/300:

* The classical bottleneck is BUILDING THE WHOLE LIBRARY, not reducing. Solving
  one path is cheap; enumerating every row for a sector group is what costs
  19 GB.
* **Greedy needs no closure at all** — it enumerates IBP actions directly, and
  the closure exists only for truth RECORDING. Every one of the 21,761 targets
  the batched classical solver could not reach is reducible by the model in
  ~1 s. The expensive classical step is now only needed to make TRAINING DATA.

A hard AND clean benchmark still does not exist. Filter on reduction difficulty
(e.g. starting weight, or observed greedy path length) rather than closure cost.

---

As of 2026-09-03 there were **two**. Both reach 100/100 on
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
