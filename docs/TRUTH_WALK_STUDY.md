# Truth-Walk Model Scoring Study

**What it answers:** for a specific integral, how does the model score the
states it was *trained* on — as opposed to the states the beam wanders into?

This is the instrument that separates a **policy** problem from a **search**
problem. Aggregate validation metrics cannot do that. When the p101 beam
campaign could not solve `1,0,1,0,0,0,5,1,1,1,...`, this study walked its
222-step truth path and found:

```
steps where a correct action ranked #1 : 211/222 (95.0%)
best-correct rank: median 1, max 4
steps where best correct is inside top-20 : 222/222
```

The model was near-perfect **on** the truth path while ranking the reference
action **50th** on beam-wandered states. That is an off-manifold generalization
failure, not a bad policy — and no aggregate metric would have told you.

---

## The four stages

Each stage feeds the next. Nothing here is optional.

| # | stage | script | output |
|---|---|---|---|
| 1 | group file | `closures/make_groups.py --targets` | `groups/group_*.json` |
| 2 | closure | `closures/batch_worker_p101.sh` | `<outdir>/out/<tag>.json` |
| 3 | truth walk + training rows | `p101_cull_corpus/worker.sh` | `out/`, `train/` |
| 4 | model scoring | `training/walk_truth_reduction.py` | per-step rank table |

### Stage 1 — group file

You do NOT hand-write these. `make_groups.py` takes a plain one-integral-per-line
list and emits the per-sector JSON groups:

```bash
python results/truth/closures/make_groups.py \
  --targets targets.txt --outdir groups/ --budget 500000 --no-merge
```

`--no-merge` keeps peak RSS ~2.3 GB instead of 12-24 GB. The emitted JSON is:

```json
{"sector": 339, "rmax": 11, "smax": 4, "seeds": 462462,
 "targets": [[3,1,0,-1,3,0,2,0,1,-1,0,0,0,-1,0], ...]}
```

Targets in one group share a sector; `seeds(S, rm, sm)` in `make_groups.py`
computes the seed count from the sector and the r/s bounds.

### Stage 2 — closure

**Use `batch_worker_p101.sh`, never `closures/worker.sh`.**

`closures/worker.sh` does not set `SAILIR_PRIME`, so it falls back to **1009**.
That is not a coefficient-only difference. From the p101 worker's own banner:

> pivots are `min(eq, key=tkey)` over the surviving support, so a different
> prime can mask a different term and pick a **DIFFERENT pivot**. These
> trajectories are therefore not just 1009's with new coefficients.

The p101 worker sets:

```bash
export SAILIR_PRIME=101                     # read by truth_engine AT IMPORT
export SAILIR_KEEP_SYSTEMS=1 SAILIR_NO_SYSTEM_CACHE=1
timeout 14400 ... batch_closure.py --group G --outdir D --budget 500000 --cap-mb 9000
```

`SAILIR_PRIME` is read by `truth_engine` at import and propagated to `ibp_env`
via `TruthEngine.__init__`, so the ENTIRE elimination runs at 101.

### Stage 3 — truth walk and training rows

`results/truth/p101_cull_corpus/worker.sh <integral>` — **the same worker that
generated the p=101 corpus** (`--prime 101`, K=1000 truthcull recipe). That is
the alignment guarantee: the rows are produced by the corpus pipeline itself,
so "how does the model rank the truth action" measures the policy and not a
schema mismatch.

```bash
SAILIR_CLOSURE_DIR=<dir holding <tag>.json>
SAILIR_WALK_OUTDIR=<where out/, train/, hard_memcap.txt go>
SAILIR_MEM_CAP_MB=7600          # cluster policy is 4 GB/core
```

Do **not** substitute `truth_record_training_v2.py`: it hardcodes `P = 1009`
with no env override, so it silently records the wrong encoding.
`truth_record_variant.py` does honour `SAILIR_PRIME`, but its own docstring
says do NOT use it for corpus generation — it is for closure dumps only.

### Stage 4 — model scoring

```bash
training/walk_truth_reduction.py \
  --topology topology_input/gravity3L \
  --shards_dir <shard dir> \
  --checkpoint p101=checkpoints/gravity3L_p101_scratch/best_model.pt \
  --n_scan_shards 1 --device cpu
```

Pack the shard with **`--val-every 1`** so every step stays in val and in step
order. The script groups by `start_target_integrals` and walks the LONGEST
trajectory, so a shard holding exactly one trajectory makes "longest"
unambiguous.

Per step it reports: how many legal actions there were, how many were correct,
the rank of the best correct action, how many wrong actions outrank it, the
model's confidence on the best correct vs the best wrong, and the ranks of all
correct actions.

---

## Traps that have actually cost time

**The prime.** Three separate scripts default to 1009 while every p101 artifact
needs 101. Two of them take `SAILIR_PRIME`; `truth_record_training_v2.py` takes
nothing. Always confirm `IBP environment using PRIME = 101` in the log before
trusting any output.

**Tag injectivity.** The tag is `tr ',' '_'` and must NOT strip `-`. Stripping
it maps distinct integrals onto one tag — 1,377 collisions measured in an 18k
sample, 90.2% of which carry negatives — so they share a closure file and
overwrite each other's training rows. Harmless in the old dots-only corpus (no
negatives), which is why it survived unnoticed.

**Closures are not free.** Check `closures/v4_p101_18k/out/` (12,584 closures
from the p=101 campaign) before building: `<tag>.json` may already exist. Top
level integrals are typically NOT in that batch.

**Cost scales hard with dot level.** Observed while building one L9 system:
r≤10 1.5s, r≤11 20.7s, r≤12 172.4s — roughly 10x per level. Budget hours, not
minutes, for r≥13, and note the 4-hour `timeout` and 9 GB cap in the worker.

**Off-manifold is the point.** `SAILIR_CONF_PROBE=1` in `beam_search_v7/v9`
logs the top model score per task BEFORE beam selection, so the sort cannot
bias the sample; that is the off-manifold counterpart to this on-manifold walk.
**It was stripped out of `greedy_reduce.py`** — only a comment remains — so the
certified greedy build cannot run the probe. Use v9 for it.

---

## Prior runs

| study | target | result |
|---|---|---|
| `results/truth/patho_walk` | `1,0,1,0,0,0,5,1,1,1,...` | rank #1 at 95.0%, top-20 at 222/222 |
| `results/truth/patho_walk_nocorrupt` | same, no corruption | `count_better.py`, `hiccup_detector.py`, `logprob_race.py` |

`patho_closure` exists because that target was not in the 18k batch — the same
situation as any new frontier integral.

---

## Open items

- The L9r9 frontier integrals of `g1023_greedy_certified` are the pending
  application: 3 of the 6 were killed by the 15 GB memory guard and are never
  retried, so that campaign cannot complete without them.
