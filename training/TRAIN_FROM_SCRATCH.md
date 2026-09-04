# Training the ftcull model FROM SCRATCH on the p=101 corpus

This branch (`p101-scratch-training`) carries the minimum needed to train
off-site. It is the `ftcull` recipe that produced the first model to solve all
125 test integrals, with ONE deliberate change: **no pretrained initialisation.**

## What is in this branch

| file | role |
|---|---|
| `training/train_classifier.py` | the trainer |
| `training/sharded_dataset.py` | sharded loader (`--shards_dir`) |
| `sailir/classifier.py` | full model |
| `sailir/classifier_nosubs.py` | `nosubs` variant — **this is the one used** |
| `sailir/ibp_env.py` | weight/order definitions |
| `sailir/topology.py` | reads the topology dir |
| `data-gen/pack_shard.py` | jsonl -> shards (one Condor job per shard) |
| `data-gen/preprocess_to_tensors.py` | tensor packing used by the above |
| `topology_input/gravity3L/` | `integralfamilies.yaml`, `kinematics.yaml`, `IBP`, `LI` (32 KB total) |

NOT in the branch, by design: the beam-search / reduction code (not needed to
train), any `.pt` weights (50-60 MB each, and we are not resuming), the corpora,
and the 79 GB of Kira artifacts under `topology_input/gravity3L/` — only the
four small files above are read.

## The corpus moves out of band

Git does not carry it. Transfer the packed shards directly:

```
<shards_dir>/shard_<K>/{train,val}.pt      # produced by pack_shard.py
```

Packing (run wherever the jsonl lives; one job per shard):

```bash
python data-gen/pack_shard.py \
  --topology topology_input/gravity3L \
  --recdir <dir of rec_*.jsonl> --outdir <shards_dir> \
  --shard K --nshards N          # [--val-every 10] [--keep-subs]
```

Two properties that matter and are easy to break:

* The split is by FILE (whole trajectory), not by sample. A global sample
  shuffle cuts trajectories in half — measured, 756 of 890 trajectories in one
  shard's `train.pt` were missing steps that had landed in val. That is also
  leakage: steps from one reduction on both sides.
* Shard assignment is `crc32` of the filename, NOT Python's `hash()`, which is
  salted per process and would give a different split in every job.

`--keep-subs` is OFF by default and should stay off for `nosubs`: the model
deletes every `sub_*` tensor on the first line of `forward()`, yet they are
44.6% of packed bytes and the loader re-reads them every epoch.

## The command

```bash
python training/train_classifier.py \
  --shards_dir   <shards_dir> \
  --model_variant nosubs \
  --prime        101 \
  --lr           1e-4 \
  --batch_size   64 \
  --token_budget 131072 \
  --epochs       200 \
  --select_on    val_loss \
  --checkpoint_every 1 \
  --num_workers  4 \
  --buffer_shards 6 \
  --output_dir   checkpoints/gravity3L_p101_scratch
```

### What changed from the ftcull recipe, and why

```
  --resume checkpoints/gravity3L_dots_v7/init.pt   REMOVED  (from scratch)
  --backbone_lr 1e-5                               REMOVED
  --prime 1009                                  -> 101
```

`--backbone_lr` splits the learning rate so a *pretrained* backbone drifts
slowly while the head adapts. With no pretrained backbone there is nothing to
protect, and its default of `0.0` means "one LR for everything" — so simply
omit it. Leaving it at 1e-5 would train the backbone 10x too slowly.

`--lr 1e-4` is not invented: it is what the existing from-scratch run used
(`checkpoints/gravity3L_dots_scratch/epochs.tsv` shows 9.998e-05 at epoch 1,
decaying). The ftcull fine-tune used 1e-5 because it was fine-tuning.

`--prime 101` MUST match the corpus. The p=101 corpus was generated with the
entire elimination at 101, pivots included — pivots are `min(eq, key=tkey)` over
the surviving support, so a different prime can mask a different term and select
a different pivot. These are not p=1009 trajectories with new coefficients.

### Epochs — TRAIN LONG

The old from-scratch run peaked at epoch 19 of 102 on the dots-only corpus
(203,728 rows). That is a small-corpus artifact. This corpus is ~10x larger, so
the overfitting onset moves out by roughly the same factor: set `--epochs` to
200+. Undertraining is the risk here, not overfitting.

Keep `--select_on val_loss --checkpoint_every 1` — costs nothing, keeps the best
checkpoint whenever the curve does turn.

Reference points for comparison, both on the OLD corpus:
* from scratch (`dots_scratch`): best val_loss **1.3205** @ep19
* ftcull fine-tune: best val_loss **0.5400** / top1 0.8677 @ep19

A from-scratch run starting from a much larger corpus should be compared against
the 1.3205 figure, not the 0.5400 one — the latter had a large unscramble
pretrain behind it.

## Checkpoint selection

Select on `val_loss`, NOT `val_top20`: `val_top20` previously picked a
checkpoint that then FAILED the beam campaign while the `val_loss` minimum
succeeded. And val_loss across different K is NOT comparable — a K=100 problem
has a lower loss floor than K=1000. This corpus is K=1000.

Ultimately the metric that decides is beam solve behaviour on the 125 test
integrals, not val_loss. The eqact T=10 model beat the baseline on val_loss
(0.5242 vs 0.5400) and still lost on wall-clock.
