# MILESTONE — first configuration to solve all 125 test integrals (2026-09-01)

**250/250 runs succeeded: 100/100 on test100 and 25/25 on test25, at BOTH beam
widths (40 and 80), with ZERO `STUCK`.**

No earlier configuration finished the full set. The previous best-known state of
the art, `ftcull4`, was killed after two-plus days with one test100 integral
never finishing.

> **UPDATE 2026-09-03** — a SECOND configuration has now finished all 125:
> `eqact T=10` (prob40, 100/100 and 25/25, zero STUCK). It is ~1.96x slower on
> test100 and ~1.10x slower on test25, so this config remains the faster of the
> two. Full side-by-side, including the per-step decomposition that explains
> the gap, is in **`FULL_125_SOLVERS.md`** — the registry of every config that
> has solved the full set.

---

## The model

| | |
|---|---|
| checkpoint | `checkpoints/gravity3L_dots_ftcull_corrupt/best_model.pt` |
| epoch | **19** (selected by `--select_on val_loss`) |
| val_loss / val_top1 | **0.539996 / 0.8677** |
| variant | `nosubs` (`IBPActionClassifierNoSubs`) — NO equation branch, NO start-target channel |
| params | 5,126,153 |

Trained by `ft_cull_corrupt_worker.sh`:

```
--shards_dir  results/truth/finetune/ftcull1848_corrupt_shards
--resume      checkpoints/gravity3L_dots_v7/init.pt      # big unscramble pretrain
--model_variant nosubs   --prime 1009
--backbone_lr 1e-5  --lr 1e-4  --batch_size 64  --token_budget 131072
--epochs 60  --select_on val_loss  --checkpoint_every 1
--num_workers 4  --buffer_shards 6
```

## The corpus

`ftcull1848_corrupt_shards` — 1843/1848 targets, 203,728 rows.

* **Cull K=1000**, upstream admissibility + fastmaxw ranking. This is NOT an
  uncalled corpus: actions/row are median ~360, mean ~545, hard max **1000**,
  with ~37% of rows sitting exactly at the cap. (The uncalled corpus is the
  older `corr13`, at ~12,700 actions/row.)
* Corruption: `SAILIR_CORRUPT_FRAC=0.2`, 2-4 scrambles per event, seed 12345 —
  the walk is knocked off the truth path and has to recover, so the model sees
  states reached by being WRONG.
* Validated before use: the 60-target cull study re-run with corruption solved
  57/60, the SAME 3 failures as uncorrupted, with ZERO `StopIteration` across
  36,123 corrupted steps.

## The beam (`beamexp_worker.sh`)

```
SAILIR_BEAM_SORT=prob          SAILIR_SCORE=decay
SAILIR_SCORE_DECAY=0.8         SAILIR_NM_PENALTY=0.1
SAILIR_TOP_K=20                SAILIR_MAX_ACTIONS=1000     # matches training K
SAILIR_V9_CULL=1               SAILIR_V9_UPENUM=1
SAILIR_TOPOLOGY=gravity3L      SAILIR_SECTOR_RANK=1
--iraws-keep-first 1000000  --beam_width {40,80}  --max_steps 50000  --prime 1009
```

TRAIN/INFERENCE ALIGNMENT IS THE POINT. `beam_search_v9.py` applies the same
upstream fastmaxw cull that generated the corpus, so the top-1000 the beam hands
the model at search time is the same action space it was fitted on.

## Results

| set | arm | solved | median | p90 | max |
|---|---|---|---|---|---|
| test100 | prob40 | 100/100 | 15.3 s | 412 s | 35,394 s |
| test100 | prob80 | 100/100 | 16.6 s | 588 s | 33,769 s |
| test25 | prob40 | 25/25 | 6,333 s | 57,573 s | 117,522 s |
| test25 | prob80 | 25/25 | 7,416 s | 61,329 s | 120,001 s |

test25 is the hard set — median solve is ~1.8 h and the worst took ~33 h.
Beam width 40 and 80 both solve everything; width is not the discriminator.

## Reproduce

```
cd results/truth/finetune
condor_submit beamexp100.sub     # 100 targets x {prob40, prob80}
condor_submit beamexp25.sub      #  25 targets x {prob40, prob80}
```

**Both .sub files now set `environment = "EXPDIR=<dir>"`, and the workers REFUSE
to run without it (exit 78).** As originally shipped, `beamexp25.sub` had no
EXPDIR line, so the worker fell back to a default of `beamexp` and wrote its
RESULTS into `beamexp/` while the .sub sent its LOGS to `beamexp25/`. The two
paths were set independently and nothing checked they agreed.

Consequence, still visible on disk: this run's prob40 outputs are SPLIT across
two trees. The 12 targets solved before 08-31 11:54 are in `beamexp/prob40/out`;
the 13 solved after are in `beamexp25/prob40/out`, because `redo_beamexp25.sub`
added the missing line mid-flight and relaunched only the unfinished targets.
`beamexp/prob40` is therefore a MIXTURE of two campaigns.

This is exactly why the counting rule below is not optional: a per-directory
`.pkl` count reports 13/25 for a run that genuinely solved 25/25.

Count successes with the TERMINAL marker only — `[v7-worker] SUCCESS in` —
never `SUCCESS_TOTAL=True`, which appears in the startup banner and has caused
false "100%" reports more than once.

## What this milestone is NOT

* Not the equation model. Every `eqact` variant tried (coefficient table,
  partial/full encoder tying, start target, T=5) plateaued ABOVE this baseline.
  `T=10` is the first to beat it on val_loss (0.5242 @ep8 vs 0.5400 @ep19) but
  has NOT yet been run on the 125 targets.
* Not K=100. The K=100 model (`ftcull_corrupt_K100`, ep19, val_loss 0.4752) is
  a separate line; its beam campaign was still short of the full set when this
  was written. Its val_loss is NOT comparable — a 100-way problem has a lower
  loss floor than a 1000-way one.
