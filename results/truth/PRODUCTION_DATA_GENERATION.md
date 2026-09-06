# Production training-data generation — the K=1000 corrupted cull recipe

**This is the ONLY sanctioned recipe for generating SAILIR fine-tuning data.**
It is the configuration that produced `ftcull1848_corrupt_shards`, the corpus
behind the first model to solve all 125 test integrals
(`finetune/MILESTONE_FULL_125_SOLVE.md`, `finetune/FULL_125_SOLVERS.md`).

Everything below is transcribed from the scripts that actually ran:

| role | file |
|---|---|
| submit (corrupted corpus) | `results/truth/truthcull/ftgen1848_corrupt.sub` |
| submit (clean corpus) | `results/truth/truthcull/ftgen1848.sub` |
| driver, sets all env | `results/truth/truthcull/ab_worker.sh` |
| thin wrapper (clean only) | `results/truth/truthcull/ftgen1848_worker.sh` |
| the program | `reduction/onestep_worker_truthcull.py` |
| the cull/record logic | `reduction/beam_search_truthcull.py` |

Do not reconstruct this from `results/truth/corrupt_corpus/` or
`reduction/truth_record_variant.py`. Those produced the **pre-cull** `corr13`
corpus, which the milestone doc records as the WORSE line. See PITFALL 1.

---

## 0. What the recipe produces, and why each piece is there

One Condor job per target. Each job replays that target's truth reduction as a
one-step reduction, and at every step writes a training row whose
`valid_actions` list **is** the culled top-K action list the deployed worker
will hand the model at inference — in rank order, with `chosen_action_idx`
indexing into it.

That alignment is the entire point. `beam_search_v9.py` applies the same
upstream/fastmaxw cull at search time, so the model is fitted on exactly the
action space it will later be searched in. A corpus recorded over the full
enumerated action list (median 303, max 34,660 actions/row) trains the model in
a space it never meets at inference.

The walk is NOT model-driven. `--model-checkpoint /dev/null` is correct and
deliberate: `SAILIR_ACTION_SELECT=truthminnew` yields exactly ONE action per
step, so no model is ever consulted. Do not supply a checkpoint.

---

## 1. PHASE 1 — the closure library (the expensive phase)

Every target needs its truth closure — the state-independent SET of
`(op, seed)` rows — as JSON:

```json
{"integral": [2,3,3,1,2,1,0,0,0,0,0,0,0,0,0],
 "sector": 63,
 "closure": [[op, [seed...]], ...],
 "pivots":  [[...], ...]}
```

`beam_search_truthcull.py` loads it read-only via `SAILIR_TRUTH_CLOSURE` and
parses `d['closure']` as `[[op, seed], ...]` (line ~2320). Files produced by
`SAILIR_DUMP_CLOSURE` or by `results/truth/closures/batch_closure.py` are
already in this format.

**Closures are recorder-independent — build them ONCE and reuse across every
corpus variant.** They cost 714.7 CPU-h for 12,584 targets and 100% of that is
the FIRST target in each group building the shared linear system (median 149 s);
every later target in the group rides it for 0.1 s. Per-target regeneration is
~4,567 CPU-h, 6.4x more, for identical output
(`results/truth/closures/closure_cost.py`).

If a target has no closure it CANNOT be walked. Those targets are the held-out
hard set (`hard_targets_p101.txt`), not corpus.

### 1.1 Sample targets

```bash
python data-gen/sample_g1023_targets.py \
  --n 59300 --seed <new seed> \
  --exclude <corpus targets>.txt --exclude <held-out>.txt \
  --out results/truth/closures/targets_<N>.txt --verify 200
```

Source pool: `results/gr_reduce/g1023/work/results` — **143,246 files, 143,190
distinct integrals**. (Other reductions exist but are far smaller: g1017 7,039,
g893 2,832, g127 2,231, g885 1,587.)

`--exclude` is MANDATORY when extending an existing corpus. Re-running with a
larger `--n` does NOT produce a superset — `rng.sample` reshuffles for every n.
Exclude the existing corpus AND any held-out set; a held-out target reappearing
in training destroys the only clean measurement of it.

`--verify 200` reopens sampled pkls and confirms `success` and
`original_integral` agree. It has never failed; if it does, stop.

Expect (measured on both 18k and 59.3k draws): 90.1% carry negative indices,
r median 9 / max 12, s median 2 / max 5, 8 of 15 indices nonzero.

### 1.2 Group the targets

```bash
python results/truth/closures/make_groups.py \
  --targets <targets>.txt --outdir <groups_dir> --budget 500000 [--no-merge]
```

One job per group; each group shares one `(sector, rmax, smax)` system, built
ONCE and read by all its targets. Manifest shape:
`{sector, rmax, smax, seeds, targets:[...]}`.

**THE MERGE DECISION — this is what determined our 30% loss.** By default groups
within a sector are merged into the largest box that fits the budget. Merging
raises each sector to its MAXIMAL box and **is what drove peak RSS to 12–24 GB**.
`--no-merge` keeps boxes far smaller (median 19,800 seeds ≈ 0.8 GB) at 1.67x
more builds.

Measured consequence of merging on the 18k campaign: **350 groups (5,398
targets, 30%) died on the memory ceiling before writing a single closure** — a
dead group takes its entire target list with it. 549 group-attempts breached
9 GB, 265 breached 19 GB. Worst surviving system: 462,462 seeds.

Under the 4 GB/core policy (1.3b), prefer `--no-merge`: 1.67x more builds is cheaper than
losing 30% of targets and re-running them.

### 1.3 Run the campaign

Worker (`batch_worker_p101.sh`), args `<group> <outdir> [cap_mb]`:

```bash
export SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1 PYTHONUNBUFFERED=1
export SAILIR_KEEP_SYSTEMS=1 SAILIR_NO_SYSTEM_CACHE=1
export SAILIR_PRIME=101
timeout 14400 <RL_MIR_IBP python> -u results/truth/closures/batch_closure.py \
  --group "$1" --outdir "$2" --budget 500000 --cap-mb "${3:-9000}"
```

`SAILIR_PRIME` is read by `truth_engine` at import and propagated to `ibp_env`
via `TruthEngine.__init__`, so the ENTIRE elimination runs at that prime —
pivots included. This matters: pivots are `min(eq, key=tkey)` over the surviving
support, so a different prime can mask a different term and pick a DIFFERENT
pivot. p=101 trajectories are NOT p=1009's with new coefficients.

Exit codes are distinguished on purpose — they need different remedies:

| exit | meaning | remedy |
|---|---|---|
| 42 | hit `--cap-mb`, aborted for retry | more memory per CPU |
| 124 | hit the `timeout` wall | longer wall, same memory |
| 0 | done | — |

Condor sizing MUST satisfy the 4 GB/core rule in 1.3b. What the 59.3k campaign
actually ran is shown for the record, but it VIOLATED that rule and got 3 jobs
reaped -- use the compliant column:

| tier | ran (violating) | USE INSTEAD | cap | wall |
|---|---|---|---|---|
| 1 capped (42) | 1 cpu / 10 GB | **1 cpu / 4 GB** | 3600 | 14400 |
| 2 capped (42) | 2 cpu / 20 GB | **5 cpu / 20 GB** | 19000 | 14400 |
| timeout (124) | 1 cpu / 10 GB | **1 cpu / 4 GB** | 3600 | 43200 |

Split failures by exit code and give each its own remedy: 42 needs more memory
(and proportionally more CPUs), 124 needs a longer wall at the SAME memory.

`p101_tier2.sub` and `p101_long.sub` are those two, queueing `grp` from
`p101_capped_groups.txt` / `p101_timeout_groups.txt`.

**SUBMIT TIER 2 WHILE TIER 1 IS STILL RUNNING.** Capped groups are settled
failures and are independent of the groups still in flight, so the two overlap
freely. Waiting for tier 1 to drain first wastes hours.

**SIZE TIER 1 FOR THE MERGE SETTING — a throughput decision, not just safety.**
The 9000 MB cap / 10 GB request above was sized for MERGED groups (RSS median
3,522 MB, max 18,934 MB). With `--no-merge` the same workload uses **median
2,321 MB, max 2,392 MB** among groups that SUCCEED (Condor's accounting tops out
~3,100 MB), so a 10 GB reservation is ~4x oversized for the common case.

That costs parallelism directly, because this pool uses PARTITIONABLE slots:
concurrent jobs = `min(free_cpus, free_mem / request_memory)` summed over the
partitionable PARENTS, not a count of free slots. Measured live with 677 free
CPUs and 543.6 GB free:

```
request_memory   concurrent jobs that fit
   10,000 MB            29      <- what we ran
    5,000 MB            80
    4,000 MB           101
    3,000 MB           144
```

For a `--no-merge` campaign set tier 1 to **request 5 GB / `--cap-mb` 4600**
BEFORE submitting. Keep them COUPLED: the cap must sit below the reservation so
an over-large group exits 42 onto the clean retry list instead of being held or
killed by Condor. Changing one without the other on a live queue breaks that
path -- which is why this belongs in the .sub, not a `condor_qedit`.

COUNTING THE CLUSTER: `condor_status` lists partitionable PARENTS as slots, so
"37 unclaimed" can mean 677 free CPUs. Use `DetectedCpus` (1,544 here across 35
machines) and the formula above, never the parent count.

**THE CAP RATE LOOKS TERRIBLE EARLY — IT IS FRONT-LOADING, NOT FAILURE.** Groups
are processed roughly largest-first, so the memory cap bites hardest at the
start. Measured on the 59.3k campaign:

```
done=43   CAP=3    (7%)
done=113  CAP=46   (41%)   <- panic point
done=2827 CAP=541  (19.1%) <- the true rate
```

Diagnose by comparing the SEED COUNTS of capped vs successful groups: if both
cohorts sit far above the overall median (here 60k-108k against 6,160), the run
is still in the big tail and the rate will fall. Set any CAP alarm well above
the known rate -- 20% against a real 19.1% fires constantly on expected
behaviour and trains you to ignore it.

### 1.3b CLUSTER MEMORY POLICY — 4 GB PER CORE, HARD

The sysadmin's limit is **<= 4 GB of memory per requested CORE**. Exceeding it
gets jobs killed by the out-of-memory reaper AND can wedge mount points on the
batch nodes, which damages OTHER users' jobs and sometimes needs manual repair.
This is not a soft budget.

We violated it on the 59.3k campaign: tier 1 ran 10 GB / 1 CPU and tier 2 ran
20 GB / 2 CPUs -- both 10 GB per core, 2.5x over. 3 jobs were reaped (exit 137).

Comply by scaling CPUs WITH memory, not memory alone:

```
  request_memory / request_cpus  <=  4 GB      ALWAYS
     4 GB  -> 1 cpu      12 GB -> 3 cpus
     8 GB  -> 2 cpus     20 GB -> 5 cpus
```

Measured need with `--no-merge` is median 2,321 MB / max 2,392 MB, so
**tier 1 = request_memory 4GB, request_cpus 1, `--cap-mb` 3600** is both
compliant and ~3.5x more parallel than the 10 GB request we used (101
concurrent jobs vs 29). Tier 2 = 20 GB needs **5 CPUs**, not 2.

### 1.3c EVERY FAILED GROUP'S TARGETS GO TO THE HARD-TARGET SET

MANDATORY, not optional. After the LAST retry tier, every target still without a
closure is held out and recorded -- never silently dropped. They are the set the
trained model is later asked to solve, and beam success is self-verifying (the
bucket drains or the walk goes STUCK), so they are scorable without truth.

Count by EXIT CODE, because they mean different things and only some are
retryable:

```
   0    ok
  42    our --cap-mb            -> retryable at more memory (with more CPUs)
 124    our `timeout` wall      -> retryable at a longer wall, SAME memory
 137    SIGKILL, external OOM   -> WE BROKE THE 4 GB/CORE RULE. Fix the request,
                                   do not just retry.
```

A grep for only `exit=0` and `exit=42` silently loses the 124s and 137s -- that
happened here (31 wall + 3 reaped were missing from the first summary).

59.3k campaign result: 364 failed tier-2 groups -> **16,361 targets** with no
closure (14,539 cap / 1,782 wall / 40 reaped), written to
`hard_targets_p101_v5.txt`. Emit with `save_hard_targets.py`, and verify the
held-out set is EXCLUDED from any later sampling (`--exclude`).

### 1.4 What to expect, from the 18k run

```
requested                 18,000 targets in 2,317 groups
solved                    12,602  (ok 12,584, failed 18)
never attempted            5,398  (350 groups died, ALL producing nothing)
cause of death             349 memory cap (exit 42), 1 SIGKILL
per-target solve seconds   median 0.13   p90 33.8   max 14,218 (3.95 h)
group peak RSS             median 3,522 MB   max 18,934 MB
disk                       329 MB closures + 43 MB logs
```

**Success rate 69.9% with merging.** Budget the sample accordingly, or use
`--no-merge` and expect better.

Accounting scripts, already written — do not rewrite them:
`p101_status.py` (coverage), `p101_dead_groups.py` (cause of death),
`closure_cost.py` (CPU-h and the batching effect),
`save_hard_targets.py` (emit the held-out set).

---

### 1.5 Sizing a corpus — measured yields, use these, do not re-derive

Per-target yield on general-index targets from g1023, p=101, K=1000 cull:

```
rows per target        39.0        (12,584 targets -> ~490,000+ rows)
disk per target        1.11 MB     (train/ jsonl)
closure disk/target    ~26 KB
```

Reference for scale: `ftcull1848_corrupt` = 210,656 raw rows / 1,848 targets =
114 rows/target. **Ours is 3x lower per target and that is expected**, not a
defect — see section 9. Multiply targets, not rows-per-target.

To hit N x the reference (203,728 packed rows):

```
targets_with_closures = N * 203728 / 39.0
targets_to_SAMPLE     = targets_with_closures / <closure success rate>
```

e.g. 10x -> ~53,900 with closures -> ~59,300 sampled at 69.9%.

**PROJECTIONS FROM A PARTIAL RUN COME IN LOW — DISCOUNT THEM.** Short walks
finish first, so any mid-run estimate understates. Bin-correcting by closure
length fixes the population mix but NOT the bias inside a bin (the largest
targets in each bin are still running). Measured error on this campaign:
samples projected 42% low, disk 25% low, and the row projection was exceeded
before the queue drained. Treat mid-run numbers as a FLOOR.

Disk at 4.3x scale (~54,000 targets), scaled from actuals:

```
closure out/ ~1.1 GB   closure logs ~150 MB
train/       ~65 GB    walk out/ ~330 MB   walk logs ~930 MB
shards       ~18 GB    (0.28x of train/, measured on the reference)
TOTAL        ~85 GB
```

Packing makes `train/` redundant; dropping it after a verified pack cuts
steady state from ~85 GB to ~19 GB.

---

## 2. The tag rule — READ THIS BEFORE WRITING ANY WORKER

```bash
TAG=$(echo "$1" | tr ',' '_')          # CORRECT
TAG=$(echo "$1" | tr ',' '_' | tr -d '-')   # WRONG — SILENT DATA LOSS
```

`tr -d '-'` deletes minus signs rather than encoding them, so distinct integrals
collide onto one tag:

```
0,1,1,0,1,1,-1,3,1,1,...  ->  0_1_1_0_1_1_1_3_1_1_...
0,1,1,0,1,1,1,3,-1,1,...  ->  0_1_1_0_1_1_1_3_1_1_...   SAME TAG
```

Colliding targets share a closure file and overwrite each other's training rows.
Measured on the real 18,000-target sample: **1,377 collisions**. On our 12,584
p=101 targets: **1,024 collisions** (11,560 distinct tags instead of 12,584).

This was harmless for the old dots-only corpora because they contain no negative
indices, so the strip was a no-op — which is exactly why the bug survived and
why copying a dots-era worker reintroduces it. `eqact_dataset.tag_of()` never
stripped, so the un-stripped form is also the one the trainer expects.

Detection: `ok` counts in the logs exceeding output files on disk. If those two
disagree, suspect collisions before anything else.

---

## 3. The exact invocation

### 3.1 Environment set by `ab_worker.sh`

```bash
export SAILIR_TOPOLOGY=gravity3L
export SAILIR_SECTOR_RANK=1
export SAILIR_ACTION_SELECT=truthminnew   # enumerate -> rank -> cut to
                                          # SAILIR_MAX_ACTIONS -> keep only
                                          # UNUSED CLOSURE rows -> take best.
                                          # REQUIRES SAILIR_TRUTH_CLOSURE or
                                          # the program exits.
export SAILIR_ACTION_SCORE=random         # truthminnew yields ONE action so
                                          # this is never consulted; `sumseed`
                                          # returns NEGATIVE scores and the
                                          # decay score logs them.
export SAILIR_RAW_EQ_CACHE_CAP=1000000000 # never evict. The default 50,000 is
                                          # SMALLER than |iraws| (~230/anchor)
                                          # once the anchor cap is off, so a
                                          # fixed-order scan misses on EVERY
                                          # lookup — cyclic thrashing. Measured
                                          # same target to step 400: t_step
                                          # 21.69s -> 1.24s, peak RSS 1038MB ->
                                          # 732MB. Results UNCHANGED, speed only.
export SAILIR_MAX_ACTIONS=1000            # THE K=1000 CULL
export SAILIR_TRUTHCULL_METRIC=upstream   # upstream admissibility + fastmaxw
export SAILIR_TRUTH_CLOSURE=<closures>/${TAG}.json
export SAILIR_TRUTHCULL_TRAIN=<traindir>/${TAG}.jsonl   # the training rows
```

### 3.2 Corruption, set by `ftgen1848_corrupt.sub` via `environment =`

```bash
METRIC=upstream
SAILIR_CORRUPT_FRAC=0.2
SAILIR_CORRUPT_MIN=2
SAILIR_CORRUPT_MAX=4
SAILIR_CORRUPT_SEED=12345
OUTSUF=_corrupt1848_f0.2_2to4
TRAINDIR=<...>/ftdata1848_corrupt_f0.2_2to4
```

Validated on the full 60-target cull study (cluster 1912908): 57/60 solved, the
SAME 3 failures as the uncorrupted study, and **ZERO `StopIteration` across
36,123 corrupted steps** — i.e. the correct action never fell outside the culled
top-1000, at scrambled states as well as truth ones. Cost 2.20x steps overall,
1.98x on the deepest quartile.

**The rate is bounded for a reason.** Each step consumes ONE closure row and
each scramble injects N, so the walk terminates only while
`FRAC * mean(N) < 1`. Here `0.2 * 3 = 0.6`. A 0.3 / 2–5 arm sits at 1.05 and
turned a 4-step target into 498 steps. Do not raise FRAC without redoing this
arithmetic.

The scramble rows are unioned INTO `_CLOSURE_SET`, the same pool the truth rows
live in, so an unscramble action is enumerated, culled and picked by identical
code. Nothing downstream can distinguish a recovery step from a truth step —
that is what makes the corpus valid.

Corruption CANNOT be faked by pre-corrupting the closure file: the closure is
loaded read-only and the walk never scrambles, so it stays on a clean path and
never needs the recovery rows. Measured three times, exactly the uncorrupted
step count.

### 3.2b Condor sizing for the WALK phase

THE RULE (1.3b): pick the memory ceiling you want, then request enough CPUs to
keep it under 4 GB per core -- `request_cpus = ceil(request_memory / 4GB)`.
Asking for more CPUs is fine; exceeding 4 GB/core is not.

```
   4 GB -> 1 cpu      8 GB -> 2 cpus     12 GB -> 3 cpus     20 GB -> 5 cpus
```

Measured over 28,621 run-1 walk jobs:

```
MemoryUsage (MB): median 352   p90 1,814   p99 2,874   max 3,861
jobs over 4,000 MB: 0
```

Production setting: **8 GB / 2 CPUs**, `SAILIR_MEM_CAP_MB=7600`. That is roughly
2x the observed max, so outliers are covered, and it is compliant. Run 1 used
10 GB / 1 CPU -- 2.5x over policy, and the kind of thing that gets jobs reaped
and wedges mount points for other users.

Always set `SAILIR_MEM_CAP_MB` BELOW the reservation so an outlier discards
itself to `SAILIR_MEM_CAP_LIST` -- joining the hard-target set per 1.3c --
instead of being killed. Run 1 recorded ZERO mem-cap discards.

The walk phase is far cheaper in memory than the closure phase because it loads
a prebuilt closure instead of building a linear system.

### 3.2c WALL-CLOCK CAP — 5 HOURS, MANDATORY

`timeout 18000` on the walk (`SAILIR_WALK_WALL` to override). Without it a
handful of targets hold the whole campaign hostage.

WHY 5h, measured on run 2's two slowest targets:

```
              rows so far   closure (= step budget)   rate      ETA
  job A          2,092              2,454           5 rows/min  ~1.2 h
  job B          2,176              3,350           2 rows/min  ~9.8 h
```

Rows accrue with steps, but the RATE DECAYS as the substitution store grows --
these two were at 2-5 rows/min after 5 hours. Finishing them meant ~10 more
hours for ~1,500 rows: **0.07% of a 2.18M-row corpus**, while blocking the pack
(which refuses to run with jobs queued). Run 2 went 40,177 of 40,179 targets in
~2 hours and then spent 4+ hours on the last two.

The closure length is the step budget, so `rows / closure_length` gives a real
completion fraction and a defensible ETA -- use it instead of guessing.

ON TIMEOUT (exit 124) the worker DELETES the partial recording. A truncated walk
is NOT the same as the contract-violation failures the reference corpus keeps
(those are complete walks with a bad end state). The target goes to the
hard-target set per 1.3c.

### 3.3 The command

```bash
PYTHONUNBUFFERED=1 \
/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python -u \
  $B/reduction/onestep_worker_truthcull.py \
  --topology $B/topology_input/gravity3L \
  --integral="$1" \
  --output $OUT/${TAG}.pkl \
  --model-checkpoint /dev/null \
  --iraws-keep-first 1000000 \
  --beam_width 1 --max_steps 20000 --prime 1009 --v7-cpus 1 2>&1
echo "EXIT=$?"
```

Note the interpreter: **`RL_MIR_IBP/conda_env/bin/python`**, not the `pyg4`
environment used elsewhere in this project.

`--iraws-keep-first 1000000` = effectively uncapped. The production default of
50 stops any substitution past the 50th from becoming an indirect-action anchor,
so from step 51 the enumerated list falls short of what the state supports. On
`2,0,0,1,0,1,0,1,1,1,0,0,0,0,0` that lost the reducing action at step 63 and the
run died STUCK at 71; uncapped, the same worker follows the truth path exactly.
See `results/truth/TRUTHCULL_DIVERGENCE.md`.

### 3.4 Condor

```
request_cpus   = 1
request_memory = 32GB
request_disk   = 4GB
queue integral from <target list, one comma-form integral per line>
```

---

## 4. Optional: memory cap and the hard-target set

```bash
export SAILIR_MEM_CAP_MB=10000              # 0/unset = no cap
export SAILIR_MEM_CAP_LIST=<path>/hard.txt  # append-only record of discards
```

A target that exceeds the cap raises `MemoryCapExceeded` and is DISCARDED, not
failed — it is written to `SAILIR_MEM_CAP_LIST` with the RSS that broke it and
deferred to the hard-target set. The point is not to avoid crashing: it is that
the corpus should contain what classical truth-closure generation can afford,
and the integrals it cannot afford become the **named held-out set** the trained
model is later asked to solve. Beam success is self-verifying (the bucket drains
or the walk goes STUCK), so those targets are scorable without ground truth.

Reference: measured peak over the 1848 dots walks was median 0 MB, p90 710 MB,
max 3,077 MB. A cap only bites on a harder distribution — which is exactly when
it is wanted.

---

## 5. Optional: separating the ACTING cull from the RECORDED cull

```bash
export SAILIR_MAX_ACTIONS=1000000000   # act UNCAPPED
export SAILIR_RECORD_K=100             # record only the top-100
```

`SAILIR_RECORD_K` caps only what is WRITTEN. When the acting action falls
outside that top-K the STEP is discarded — the row is not written — but the walk
takes the step anyway and the trajectory continues. Per-step discard, never
per-trajectory.

**Never pass `SAILIR_MAX_ACTIONS=0` to mean uncapped.** Only the two
`kept = scored[...]` lines treat 0 as uncapped; `_select_actions` does
`nv <= max_actions` and `argpartition(keys, max_actions)[:max_actions]`, so K=0
selects ZERO actions and the walk goes STUCK at step 0 (measured, pilot cluster
1917856). Pass a huge number instead.

Do not set the ACTING cull to 100 directly: `surv` comes from `kept`, so an
empty top-100 means the walk cannot proceed at all — measured, no closure row
reaches the top 100 on 84% of 400+ step trajectories.

**K=1000 is the production setting.** K=100 is a separate line
(`data-gen/derive_k100.py` slices it from K=1000 shards without re-running,
because the recorded list is already in fastmaxw rank order). Its val_loss is
NOT comparable to K=1000's — a 100-way problem has a lower loss floor.

---

## 6. Packing to shards

```bash
python data-gen/pack_shard.py \
  --topology <topo> --recdir <traindir> --outdir <shards_dir> \
  --shard K --nshards N [--val-every 10] [--keep-subs]
```

One Condor job per shard; writes `<outdir>/shard_K/{train,val}.pt`, the layout
the trainer discovers via `--shards_dir`.

### 6.1 Packing a corpus that lives in SEVERAL directories

`pack_shard.py` takes ONE `--recdir`, and a shard-packed corpus is not
bit-comparable with a piecewise one ("do not mix; repack whole"). So symlink
every run into a single directory and pack in one pass. Use
**`results/truth/combine_and_pack.sh`** -- do not do this by hand.

It enforces the three things that can silently ruin the corpus:

* **Refuses to run while any job is queued.** Packing while walks are still
  appending captures PARTIAL trajectories. This is a hard abort, not a warning.
* **Aborts if `combined != run1 + run2`.** Two runs must have disjoint
  filenames; run 2's sampling used `--exclude`, and this re-checks it at pack
  time rather than trusting it. Verified 0 overlap on the p=101 corpus.
* **Uses `find`, never a glob** -- these dirs hold 12k-40k files (PITFALL 12).

Symlinks, not copies: the combined corpus is ~30 GB.

### 6.1b THE FILES MUST BE NAMED `rec_<tag>.jsonl`

`pack_shard.py` globs for **`rec_*.jsonl`** (line ~82). The truthcull worker
writes `<tag>.jsonl` with NO prefix -- the `rec_` prefix came from the OLDER
`truth_record_variant.py` worker, so a truthcull corpus does not match it.

Symptom, if you get this wrong: every shard job exits SUCCESSFULLY (rc=0) with

```
no non-empty recordings under <recdir>
```

and writes zero shards. Measured: all 788 jobs "finished", 0 shards, and only
the non-empty `.err` files revealed it. A clean exit code makes this look like
success -- check the shard COUNT, never the queue draining.

Fix it in the SYMLINK step, not in `pack_shard.py`, which older corpora share:

```bash
find "$R1" "$R2" -name '*.jsonl' -printf '%p\0' \
  | xargs -0 -I{} sh -c 'ln -sf "$1" "$2/rec_$(basename "$1")"' _ {} "$ALL"
```

### 6.2 Shard count -- size by ROWS, not by file count

The reference packed 210,656 rows over 76 shards = **~2,772 rows/shard**. Match
that density:

```
nshards = ceil(total_rows / 2772)
```

A file-count split is wrong here: our targets average ~44 rows against the
reference's 114, so equal file counts give shards a third the weight.

### 6.3 Condor for the pack

`request_cpus=2 request_memory=8GB` (the 4 GB/core rule, 1.3b). Measured 2.99 GB
peak on a 48-file shard. Log names MUST carry `$(ClusterId)` (PITFALL 10).

After a VERIFIED pack the raw `train/` jsonl is redundant -- dropping it takes
the corpus from ~85 GB to ~19 GB.

Two properties that matter:

* **The split is by FILE (trajectory), not by sample.** The old global shuffle
  cut trajectories in half — 756 of 890 trajectories in one shard's `train.pt`
  were missing steps that had landed in val/test, which is also leakage (steps
  from one reduction on both sides).
* **Shard assignment is by `crc32` of the filename**, not Python's `hash()`,
  which is salted per process and would give a different split in every job —
  silently producing overlapping or missing shards.

A shard-packed corpus is NOT bit-comparable with a single-process pack. Do not
mix; repack whole.

`--keep-subs` is OFF by default: the `nosubs` model deletes every `sub_*` tensor
on the first line of `forward()`, yet they are 44.6% of packed bytes
(`subs_raw` alone 195 MB of a 448 MB shard) and the sharded loader re-reads and
re-unpickles them every epoch. Pass it only for the full `IBPActionClassifier`.

---

## 7. Adapting to a different prime (e.g. p=101)

Change **two** things and nothing else:

1. `--prime 101` on the `onestep_worker_truthcull.py` command line.
2. `SAILIR_TRUTH_CLOSURE` must point at closures generated at THAT prime. The
   truth engine's row selection is arithmetic, so a p=1009 closure is not
   assumed valid at p=101.

Note `truth_record_variant.py` had `P = 1009` hardcoded while `truth_engine.P`
read `SAILIR_PRIME` — the two halves of one run over different fields, silently,
since coefficients stay small integers either way. Fixed 2026-09-03; check any
new program for the same split before trusting it.

Accidental zeros are NOT a measurable concern at p=101 for raw equations:
support differences between p=1009 and p=101 measured **0 of 84,000 equations
(0.000%)** (`results/truth/closures/prime_masking.py`).

---

## 8. Verification checklist — run these BEFORE trusting a corpus

1. **Actions per row matches the cull.** MEASURED on the real corpus
   (`ftdata1848_corrupt_f0.2_2to4`, 7,265 rows sampled from its 1,848 files):

   ```
   median 375   mean 554   MAX 1000   rows exactly at 1000: 39.6%
   ```

   The hard ceiling at exactly 1000 and the ~40% pile-up ON it are the
   signature — a culled corpus cannot exceed K, and a large fraction saturates
   it. If instead you see `median 303 / mean 2,393 / max 34,660` (the measured
   profile of the discarded 2026-09-03 campaign), the cull did NOT run and you
   have a `corr13`-style uncalled corpus.
2. **Output files == successful targets.** A shortfall means tag collisions
   (section 2), not zero-sample walks.
3. **Corruption actually fired.** The STATS line carries
   `injected= events_ok= attempts= skipped_events=`. A silent no-op looks
   exactly like a pass — that is how the corruption study produced a fake
   PASS=6 FAIL=0 before anyone checked scrambling had happened.
4. **`[record-k]` counters** if `SAILIR_RECORD_K` is set: rows written vs steps
   discarded. A separation that never fires looks like a passing run.
5. **Schema matches a known-good corpus** — same key set as
   `results/truth/corrupt_corpus/out/*.jsonl` (17 keys).
6. **Count successes by the terminal marker only.** For beam runs that is
   `[v7-worker] SUCCESS in`. NEVER `.pkl` counts (retries leave extras: 107
   pkls for 100 targets) and NEVER `SUCCESS_TOTAL=True`, which appears in the
   startup banner and has produced false "100%" reports repeatedly.

---

## PITFALLS — every one of these has actually happened

**PITFALL 1 — using `truth_record_variant.py` instead of
`onestep_worker_truthcull.py`.** (2026-09-03, cost: a full 12,584-job campaign,
discarded.) `truth_record_variant.py` does not rank and does not cull; it
produced the pre-cull `corr13` corpus. The cull is not an environment variable
you can add to it — the ranking lives in `beam_search_truthcull.py`. The
resulting corpus cannot be salvaged by truncation either: `derive_k100.py` works
only because the truthcull recorder writes its list ALREADY in fastmaxw rank
order, whereas the variant writes enumeration order, so "the first 1000" is
arbitrary. The trap is that `results/truth/corrupt_corpus/worker.sh` looks like
"the ftcorrupt infra" and carries the right-looking `SAILIR_CORRUPT_*` values —
but it predates the cull entirely.

**PITFALL 2 — `tr -d '-'` in the tag.** See section 2. 1,024 silent collisions
on the p=101 target set.

**PITFALL 3 — trusting prose over the generating script.** The corruption
parameters in `MILESTONE_FULL_125_SOLVE.md` are correct, but the doc describes
the corpus, not the program; reading it as "the ftcorrupt recipe" is what led to
PITFALL 1. Always open the `.sub` and the worker it names.

**PITFALL 4 — `SAILIR_MAX_ACTIONS=0` to mean uncapped.** Selects zero actions;
STUCK at step 0. Section 5.

**PITFALL 5 — the EXPDIR split-path class of bug.** `beamexp25.sub` originally
had no `EXPDIR`, so the worker defaulted to `beamexp` and wrote RESULTS to one
tree while its LOGS went to another. Both paths were set independently and
nothing checked they agreed. Any new `.sub` must set every path its worker
reads, and the worker should REFUSE to run without them (exit 78).

**PITFALL 6 — the wrong Python.** This chain uses
`RL_MIR_IBP/conda_env/bin/python`, not `pyg4`.

**PITFALL 7 — `cd DIR && nohup ... &` with relative paths.** The shell's cwd
resets between tool calls, so the redirect target resolves somewhere else and
the job dies silently with no log. Use ABSOLUTE paths inside every launcher, and
verify the log exists with a fresh mtime within 10 s of launching.

**PITFALL 8 — reading `condor_q ... -totals | tail -2`.** That cuts off the
"Total for query" line (the one for YOUR cluster) and leaves only "Total for all
users". Reading the absence of your cluster as an empty queue reports a running
campaign as finished. Grep for `"Total for query"` explicitly.

**PITFALL 14 — NEVER INFER TEST-SET MEMBERSHIP FROM A DIRECTORY.** Result trees
are mixtures: `beamexp/` holds targets from several campaigns because of the
EXPDIR split-path bug (PITFALL 5), so "everything under beamexp/prob40 is
test25" is false.

A comparison script keyed that way reported **28 baseline solves for a
25-target set** and inflated every baseline figure it produced -- quoted three
times before the impossible denominator was challenged.

Take membership from the campaign's own jobs file and drop anything else,
whatever tree it sits in:

```python
keep = {l.split(',')[0].strip() for l in open(jobs_file) if l.strip()}
...
if tag not in keep:      # foreign target from a mixed tree
    continue
```

And ALWAYS print a known-total column as a self-check -- a completed baseline
must come back 100/100 and 25/25. If a denominator exceeds the set size, stop.
Canonical script: `finetune/p101_vs_k1000_v2.py`.

**PITFALL 13 — `cat | wc -l` IS 17x SLOWER THAN `wc -l` ON MANY FILES.**
Counting rows across the 52,761-file corpus:

```
find ... -print0 | xargs -0 cat | wc -l      25+ MINUTES   (killed at 24 min)
find ... -print0 | xargs -0 wc -l \
    | awk '$2=="total"{s+=$1} END{print s}'      89 SECONDS   <- USE THIS
```

`cat` funnels every byte through one pipe into a single `wc`; `xargs wc -l` lets
`wc` open each file directly and count in place, and `xargs` batches mean several
`total` lines to sum. Same answer (2,184,198), 17x faster.

The same applies to any whole-corpus scan. Reserve `cat` for when you actually
need the concatenated BYTES, not a count.

**PITFALL 12 — SHELL GLOBS SILENTLY BREAK ON BIG DIRECTORIES.** These corpora
hold 12k-40k files per directory, past the ARG_MAX limit, and the failure is not
an error -- it is a WRONG NUMBER. Hit three times in one day:

```
ls $D/out/*.json | wc -l        -> 0        (looked like the closures vanished)
rm -f $W/logs/*.out             -> "Argument list too long"
grep -h EXIT= $W/logs/*.out     -> read an unrelated file; failures showed 0
```

The grep case is the dangerous one: a monitor reported `bad=0` when the true
count was 23, because the glob failed to expand and grep fell back to reading
whatever was left on its command line.

ALWAYS use find for these:

```bash
find $D/out -name '*.json' | wc -l
find $W/logs -name '*.out' -print0 | xargs -0 grep -h "EXIT="
find $W/logs -type f -delete
```

**PITFALL 10 — CHUNKED SUBMISSION CLOBBERS LOGS.** Every `condor_submit` makes
a NEW cluster whose Process numbering restarts at 0, so
`output = logs/$(Process).out` means chunk 2's job 0 overwrites chunk 1's job 0.
Measured: 8 clusters, 8,292 targets completed, only 3,000 log files on disk --
and the "ok" count went DOWN between polls because the directory was being
overwritten underneath it.

Always use `logs/$(ClusterId)_$(Process).out`. The CORPUS is unaffected (train
files are named by target tag, which is unique), but log-based accounting is
worthless without this. Verify counts against `train/` file names, not logs.

**PITFALL 11 — a drip feeder's cap is a FLOOR, not a ceiling.** It checks the
queue BEFORE submitting a chunk, so the real maximum is `MAXQ + CHUNK`. With
MAXQ=6000 and CHUNK=3000 the queue peaked at 8,022, not 6,000. Size MAXQ as
`desired_ceiling - CHUNK`.

SUBMIT IN CHUNKS AT ALL: on 2026-09-04 a single 15,000-job submission left the
schedd unable to answer `condor_q` OR `condor_rm`
("SECMAN:2007: Read failure during security negotiation"), and even 1,000 jobs
jammed it while it was still recovering. `MAX_JOBS_PER_SUBMISSION` is 20,000 but
that is not a safe working figure. Use
`results/truth/p101_cull_corpus_v2/drip_submit.sh`, which holds a bounded queue
and treats an unreachable schedd as WAIT rather than as an empty queue.

**PITFALL 9 — counting anything except the TERMINAL MARKER.** All three of these
happened in one day, and all three are the same error:

| counted | why it lies |
|---|---|
| `.pkl` / output files | retries leave extras — 107 files for 100 targets |
| `condor_q \| tail -2` | see PITFALL 8 |
| `*.out` log files | created when a job STARTS, not when it finishes |

The last one fired a false "WORKER BROKEN" alarm one minute into the 59k closure
campaign: 421 logs existed, 0 closures, and every one of those jobs was healthy
and still building its system. An alarm that fires on healthy jobs gets ignored,
which is worse than no alarm.

ALWAYS count the program's own terminal marker — `[worker] exit=` for the batch
closure worker, `[v7-worker] SUCCESS in` for beam runs, `EXIT=` for ab_worker.
And never `SUCCESS_TOTAL=True`, which is in the startup banner.

---

## 9. Why this corpus yields fewer rows per target than ftcull — not a defect

Do not treat the gap as a bug and re-investigate it. It was chased to ground on
2026-09-03 and has two causes, both by design:

**Dots vs general indices.** `ftcull`/`ftcorrupt` targets are **100% dots-only**
(s=0, no numerators). g1023-sampled targets are **86% general** (s>=1, s median
1, max 4). Measured inside one campaign — same prime, same code, same day —
dots-only yields 47.7 rows/target and general yields 26.2. General indices are
~1.8x thinner.

**Curated vs representative.** The dots campaigns SELECTED for yield;
`dots_sweep_v6/select_targets.py` says so outright: *"difficulty band shifted UP
... Longer trajectories yield more samples each"*, preferring deep L>=7 sectors.
g1023 sampling is on-distribution by construction. At MATCHED (L, r) — 25 cells,
covering 1,701/1,734 of ours and 1,759/1,848 of theirs — our dots-only subset
still yields 0.71x, i.e. curation alone is worth ~1.4x.

Ruled OUT, with evidence, so nobody repeats it: the prime (raw-equation support
differs in **0 of 84,000 equations, 0.000%**, `prime_masking.py`); the weight
refactor (its two changes are order-preserving and the recorder uses `tkey`
anyway); the (L, r) mix (matching it does not move the ratio); corruption
settings (ours is STRONGER, which pushes yield UP).

The corpus is broad and shallow BY DESIGN — 6.8x the targets at ~1/3 the rows
each — because the deployed model meets general indices, and a dots-only corpus
has never seen the dominant feature of what it is asked to reduce.
