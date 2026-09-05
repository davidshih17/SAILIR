# gravity3L_p101_scratch — trained model (3-loop gravity, p=101, FROM SCRATCH)

Action classifier for the **3-loop gravity (GR) topology**, trained on the
**p=101 all-indices corpus**, with **no pretrained initialisation** — the
recipe in [`training/TRAIN_FROM_SCRATCH.md`](../../training/TRAIN_FROM_SCRATCH.md).

## ⚠️ `prime=101`, NOT 1009

Unlike every other checkpoint in this repo, this model was trained at
**`--prime 101`**. The p=101 corpus was generated with the entire elimination
at 101, pivots included, so it is not p=1009 trajectories with recomputed
coefficients. Loading this model with `prime=1009` will silently produce a
wrong coefficient encoding. Verified before training: coefficient range
[1, 100] with zero coeffs >= 101 across 20 sampled shards
(`training/run_verify_corpus.sh`).

## The checkpoint

| File | Epoch | train loss | val loss | val top-1 | val top-5 | val top-20 |
|------|------:|-----------:|---------:|----------:|----------:|-----------:|
| `best_model.pt` | 34 | 0.1478 | **0.3110** | 92.38% | 98.69% | 99.69% |

Selected on `val_loss` (`--select_on val_loss`), which is the minimum over the
whole run.

**val_loss and top-1 disagree on this run, and the gap is not noise.** The
run continued to E54, where val_loss had risen 22% to 0.3785 while val top-1
had *improved* to 92.85% and top-5 to 98.71%:

| | E34 (`best_model.pt`) | E54 (final) |
|---|---:|---:|
| val loss | **0.3110** | 0.3785 |
| val top-1 | 92.38% | **92.85%** |
| val top-5 | 98.69% | **98.71%** |
| val top-20 | **99.69%** | 99.65% |

That divergence is the confidently-wrong-sample effect: as the LR annealed the
model sharpened its distribution, which costs a lot of cross-entropy on the
~7% it ranks wrong while ranking quality kept improving. Only the val_loss
minimum is committed here, per the selection criterion; **E54 (`last.pt`) and
all 56 per-epoch checkpoints are retained on the training machine** at
`/home/shih/work/SAILIR_p101/checkpoints/gravity3L_p101_scratch/` if the beam
campaign wants the higher-top-1 model instead.

Do NOT compare this val_loss against the historical `dots_scratch` (1.3205) or
`ftcull` (0.5400) figures — those are on the old **dots-only** corpus. This one
covers **all indices**: a different action set, state distribution and loss
floor. See TRAIN_FROM_SCRATCH.md.

## How to load — `nosubs` variant

The model is `IBPActionClassifierNoSubs` from
[`sailir/classifier_nosubs.py`](../../sailir/classifier_nosubs.py) — **not**
`IBPActionClassifier`. The state_dict has no `subs_enc.*` keys.

```python
import torch
from sailir.classifier_nosubs import IBPActionClassifierNoSubs

ckpt = torch.load(path, map_location=device, weights_only=False)
assert ckpt['args']['model_variant'] == 'nosubs'
assert ckpt['args']['prime'] == 101          # <-- NOT 1009

model = IBPActionClassifierNoSubs(
    embed_dim=256, n_heads=4, n_expr_layers=2, n_cross_layers=2,
    prime=101, n_indices=15, n_denominators=10, n_ibp_ops=21,
    # dims verified against Topology.from_dir('topology_input/gravity3L')
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

All hyperparameters are also in `ckpt['args']`.

## Training provenance

- Data: `data/` on the training host — 788 shards, **1,958,595 train /
  220,996 val samples per epoch**. Packed without `--keep-subs` (no
  `subs_raw` field), as `nosubs` requires.
- Topology: `topology_input/gravity3L` (n_indices=15, n_denominators=10,
  n_actions=21)
- Params: 5,126,153
- 3× NVIDIA L40S, DDP (`torchrun --nproc_per_node=3`), batch 64/rank
  (effective 192), AdamW wd 1e-5, `--prime 101`, `--seed 0`,
  `--n_val_shards all`, `--checkpoint_every 1`
- Launcher: [`training/run_train_p101_scratch.sh`](../../training/run_train_p101_scratch.sh)
- ~80 ms/batch, ~819 s/epoch (9,928 train + 1,088 val iters per rank)

### LR schedule — changed mid-run at E24/E25

Started on the recipe's implicit `CosineAnnealingLR(T_max=200, eta_min=lr/10)`.
By E23 the LR was still at **97% of peak** (9.71e-5) while val_loss had
flattened into a 0.323–0.331 oscillation band — the anneal was ~110 epochs
away from arriving. Killed at E24 and resumed from `checkpoint_epoch24.pt`
onto `--lr_schedule plateau` (`ReduceLROnPlateau` on val_loss, factor 0.3,
patience 5, min_lr 1e-6).

| epoch | lr | val loss | note |
|---|---|---|---|
| 1–24 | 1.00e-4 → 9.68e-5 | 0.6676 → 0.3239 | cosine |
| 25–32 | 9.68e-5 | 0.3238 → 0.3255 | plateau, no cut yet |
| **33** | **2.91e-5** | 0.3249 | 1st cut |
| **34** | 2.91e-5 | **0.3110** | ← best; val_loss −4.3%, top1 +0.76pp in one epoch |
| 40 | 8.72e-6 | 0.3434 | 2nd cut — no further gain |
| 46 | 2.62e-6 | 0.3649 | 3rd cut — no gain |
| 52–54 | 1.00e-6 (floor) | 0.3765 → 0.3785 | schedule exhausted |

Only the **first** cut helped. Stopped at E54 (of a nominal 200) once the LR
hit `min_lr` and val_loss had risen monotonically for 20 epochs — the
remaining 146 epochs at 1e-6 could not have beaten E34.

The `--lr_schedule {cosine,plateau}` flag added for this run defaults to
`cosine`, so the original recipe is unchanged for other runs.

### Not yet tried

Regularisation was left at the defaults (dropout 0.1, weight_decay 1e-5). The
train/val gap grew 4.6× over the run (+0.064 at E20 → +0.296 at E52) with
train loss reaching 0.078, so there is likely headroom in: temperature
scaling / checkpoint averaging (free, no retrain), label smoothing, higher
weight decay (1e-5 is effectively off for AdamW), higher dropout, or more data.
