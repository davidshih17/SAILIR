# SAILIR

**S**elf-supervised **AI** for **L**oop **I**ntegral **R**eduction.

Code, trained model, and benchmark data accompanying:

> **Learning to Unscramble Feynman Loop Integrals with SAILIR**
> David Shih, 2026. arXiv:2604.05034.

## Topologies

Phase-2 makes the pipeline **topology-agnostic**: every script (data-gen,
preprocess, train, eval) takes a `--topology topology_input/<family>/`
argument that selects the integral family at runtime. The repo ships with
two configurations:

- `topology_input/trianglebox/` — 2-loop triangle-box (6 propagators + 1 ISP),
  the published phase-1 family. Used by all examples below unless noted.
- `topology_input/pentagonbox/` — 2-loop pentagon-box, TA family from Kira
  (8 propagators + 3 ISPs). See `topology_input/HOW_TO_DERIVE_FROM_KIRA.md`
  for how to derive a new topology's inputs.
- `topology_input/gravity3L/` — 3-loop gravity (post-Minkowskian potential)
  family, FIRE family 40 (10 propagators incl. linear/eikonal + 5 ISPs).

**Production reduction runs** use the **symmetry-enhanced general-topology
pipeline** — sector canonicalization, the sector-senior total order, canonical
masters, and a general, numerically-gated symmetry engine — documented in
[`reduction/README.md`](reduction/README.md) (§4b there is the entry point;
the examples below reproduce the paper's plain IBP+LI runs).

Each topology directory contains:

```
integralfamilies.yaml   propagators + ISPs + symmetry classes
kinematics.yaml         kinematic invariants (d, s_ij, ...) + their finite-field values
IBP                     IBP identity templates (shift, coefficient)
LI                      Lorentz-invariance identities
masters                 Kira's master basis
```

## gravity3L, p=101, all-indices (this branch)

The `p101-scratch-training` branch carries a complete second pipeline: a 3-loop
gravity model trained FROM SCRATCH on an all-indices corpus at prime 101. The
older sections below describe the trianglebox/pentagon-box phase-1 flow and
still apply to those topologies.

Three documents cover it end to end. **Read them in this order; each is the
authoritative source for its stage, and each records the measured reason behind
every setting rather than just the setting.**

| stage | document | what it covers |
|---|---|---|
| 1. data | [`results/truth/PRODUCTION_DATA_GENERATION.md`](results/truth/PRODUCTION_DATA_GENERATION.md) | sample targets -> closures -> walks -> shards. Every env var, the K=1000 cull, corruption rates, cluster limits, and **14 pitfalls that each cost real time** |
| 2. training | [`training/TRAIN_FROM_SCRATCH.md`](training/TRAIN_FROM_SCRATCH.md) | the exact command, what differs from the ftcull recipe and why, epochs, checkpoint selection |
| 3. the model | [`checkpoints/gravity3L_p101_scratch/README.md`](checkpoints/gravity3L_p101_scratch/README.md) | metrics, how to load (`nosubs` variant), and the `prime=101` warning |

### Why data generation is TWO phases

Training rows are not produced directly from an integral. There are two distinct
stages, and keeping them separate is what makes the corpus affordable.

**Phase 1 - the closure library.** For a target integral, the truth engine
solves which IBP identities are needed to reduce it, and emits them as a
*state-independent SET* of `(op, seed)` rows:

```json
{"integral": [2,3,3,1,2,1,0,0,0,0,0,0,0,0,0], "sector": 63,
 "closure": [[op, [seed...]], ...], "pivots": [[...], ...]}
```

It is a SET, not a path: consuming those rows in ANY order reaches a successful
reduction. This phase is the expensive one -- 715 CPU-h for 12,584 targets -- so
it is done in BATCHES, one shared linear system per group of targets in the same
sector. The first target in a group pays to build the system (median 149 s);
every later target in that group reads it in 0.1 s. Building per target instead
costs 6.4x more for identical output.

Closures are also **recorder-independent**: build them once and reuse them for
every corpus variant. A target with no closure cannot be walked, and belongs in
the held-out hard set rather than the corpus.

**Phase 2 - the truth walker.** Given a closure, the walker replays the
reduction one step at a time. At each step it enumerates the candidate actions,
ranks them, cuts to the top K=1000, keeps only unused closure rows, and writes a
training row: the state, that culled top-1000 list, and which entry is correct.

The cull is the whole point. `beam_search_v9.py` applies the SAME ranking at
search time, so the model is fitted on exactly the action space it will later be
searched in. A corpus recorded over the full enumerated list (up to 34,660
actions) trains it in a space it never meets at inference.

The walker also **corrupts** 20% of steps by injecting 2-4 random identities,
knocking the walk off the truth path so the corpus contains recovery states --
the case a clean recording can never contain. Those scramble rows go into the
same pool as the truth rows, so nothing downstream can tell a recovery step from
a truth step.

### The pipeline in brief

```bash
# 1. SAMPLE targets from a real reduction (--exclude is MANDATORY when extending)
python data-gen/sample_g1023_targets.py --n 59300 --seed <s> \
  --exclude <existing corpus>.txt --exclude <held-out>.txt \
  --out targets.txt --verify 200

# 2. GROUP them, then build closures (one Condor job per group).
#    --no-merge keeps peak RSS ~2.3 GB instead of 12-24 GB.
python results/truth/closures/make_groups.py \
  --targets targets.txt --outdir groups/ --budget 500000 --no-merge
condor_submit <tier-1 .sub>          # see 1.3 for the retry tiers

# 3. WALK each target -> training rows (the K=1000 cull happens here)
results/truth/p101_cull_corpus/worker.sh <integral>
#    At scale, drip-feed it -- a 40k-job submission jammed the schedd:
results/truth/p101_cull_corpus_v2/drip_submit.sh <sub> <targets> 4000 2000 60

# 4. PACK to shards (refuses to run while jobs are still writing)
results/truth/combine_and_pack.sh

# 5. TRAIN -- see TRAIN_FROM_SCRATCH.md
```

### Diagnosing WHY a target fails: the truth-walk study

When a target will not solve, this separates a POLICY problem from a SEARCH
problem — walk its truth path and score the model at every step. On the one
integral the p101 campaign never solved, the model ranked a correct action #1
at 95.0% of 222 steps and inside the top-20 at every step, while ranking the
reference action 50th on beam-wandered states: off-manifold generalization,
not a bad policy. Full recipe, and the traps (the prime, tag injectivity,
closure reuse):

    docs/TRUTH_WALK_STUDY.md

### Running the trained model on an integral

```bash
EXPDIR=<dir> results/truth/finetune/p101_beam_worker.sh <tag> prob 40
```

`<tag>` is the integral with commas as underscores (`1_3_0_1_...`). The worker
sets the beam configuration that solved the 125-target benchmark
(`BEAM_SORT=prob`, decay 0.8, NM penalty 0.1, TOP_K 20, `MAX_ACTIONS=1000`
matching the corpus cull) and calls `reduction/onestep_worker_v9.py`.

**`--prime 101` is not optional for this checkpoint.** Every other model in the
repo is p=1009; running this one at 1009 produces a wrong coefficient encoding
SILENTLY -- no error, just wrong numbers. `p101_beam_worker.sh` differs from the
p=1009 `beamexp_worker.sh` in exactly two places, the checkpoint and the prime.

`EXPDIR` is required and the worker exits 78 without it: it once defaulted, and
a run's outputs and logs silently landed in different trees for two days.

### Comparing against the baseline

```bash
python results/truth/finetune/p101_vs_k1000_v2.py
```

Reports solves and CPU-hours against the K=1000 ftcull model
([`FULL_125_SOLVERS.md`](results/truth/finetune/FULL_125_SOLVERS.md)), paired
per target. Count solves ONLY by the `[v7-worker] SUCCESS in` terminal marker --
`.pkl` counts over-report, and test-set membership comes from the jobs file, not
from which directory a log sits in.

---

## Repository layout

```
sailir/
  topology.py            Topology dataclass; init_from_topology() configures the module
  ibp_env.py             IBP environment (identities, action enumeration); topology-driven
  classifier.py          Action classifier; encoder dims taken from the topology
  symmetries.py          Sector-symmetry library (standalone, optional)
scripts/
  data_gen/              Self-supervised trajectory generation + JSONL→tensor packing
                         (now supports multi-file --input and per-worker sharding)
  train/train_classifier.py
                         Cross-entropy training with topology-aware collate
  eval/                  Hierarchical async orchestrator + single-integral worker + replay
  kira/                  Kira inputs for the comparison side of Fig. 3
  plots/                 Fig. 2 and Fig. 3
topology_input/<family>/ Per-topology config (see above)
checkpoints/best_model.pt
                         Published phase-1 trained model (trianglebox)
results/                 Benchmark CSVs and training log
```

## Requirements

- Python 3.10+
- PyTorch, NumPy, Matplotlib
- HTCondor (for §1 data-gen and §4 benchmark)
- Kira + Fermat (for §6 only)

```bash
pip install torch numpy matplotlib
```

## Quick start

Reduce a single trianglebox integral with the published checkpoint:

```bash
python scripts/eval/onestep_worker.py \
    --topology topology_input/trianglebox \
    --integral=2,1,2,1,2,2,-4 \
    --model-checkpoint checkpoints/best_model.pt \
    --output reduction.pkl \
    --beam_width 20 \
    --prime 1009 \
    --paper-masters-only -v

python scripts/eval/replay_reduction_path.py --path reduction.pkl
```

## End-to-end reproduction (trianglebox)

### 1. Generate training data

100 Condor workers × 1000 scrambles each (`--max_steps 25`). ≈18% of attempts
are discarded as `SKIPPED_VANISHING`; the remaining ≈82 000 trajectories yield
the $8\times10^4$ trajectories / $1.06\times10^6$ samples reported in paper §IV.B.

```bash
export SAILIR_DIR=$(pwd)
export PYTHON=$(which python)

bash data-gen/submit_datagen.sh 100 1000
condor_submit data-gen/datagen_job_custom.jdl
# ... wait for all 100 jobs to finish ...
bash data-gen/merge_outputs.sh data/raw_jsonl/
```

### 2. Pack JSONL into tensors

```bash
python data-gen/preprocess_to_tensors.py \
    --topology   topology_input/trianglebox \
    --input      data/raw_jsonl/multisector_training_data.jsonl \
    --output_dir data/multisector/
```

Defaults reproduce the paper's preprocessing: `--val_split 0.1`, `--test_split 0.1`, `--seed 42`. Produces `train.pt` (≈80%), `val.pt` (≈10%), `test.pt` (≈10%).

`--input` accepts multiple files now (e.g. all per-worker JSONLs) and
streams them in sequence, so no concatenation step is needed.

### 3. Train

```bash
python training/train_classifier.py \
    --topology   topology_input/trianglebox \
    --data_dir   data/multisector/ \
    --output_dir checkpoints/ \
    --epochs 30 --batch_size 256 --lr 4e-4 --prime 1009 --device cuda

python scripts/plots/plot_training_curve.py --log results/training_log/train.log
```

### 4. Reduce the 16 benchmark integrals

```bash
INTEGRALS=( "2,1,2,1,2,2,-4"  "1,1,2,2,1,3,-5"  "1,1,3,2,2,1,-6"  "2,3,1,1,2,1,-7"
            "2,2,2,1,1,3,-4"  "1,1,2,3,2,2,-5"  "1,4,2,1,2,1,-6"  "2,1,1,2,3,2,-7"
            "2,3,1,3,1,2,-4"  "1,2,2,2,1,4,-5"  "3,2,3,2,1,1,-6"  "3,1,1,1,1,5,-7"
            "2,3,3,3,1,1,-4"  "2,2,3,3,2,1,-5"  "3,2,1,3,2,2,-6"  "2,2,3,3,1,2,-7" )

mkdir -p logs results work-dir
for I in "${INTEGRALS[@]}"; do
    label=$(echo "$I" | tr ',' '_' | tr '-' 'm')
    python -u scripts/eval/hierarchical_reduction.py \
        --topology topology_input/trianglebox \
        --integral=$I --output results/reduction_${label}.pkl \
        --work-dir work-dir/${label} \
        --model-checkpoint checkpoints/best_model.pt \
        --beam_width 20 --prime 1009 --paper-masters-only --beam-sort mixed \
        > logs/async_${label}.log 2>&1 &
done
wait
```

### 5. Plot

```bash
python scripts/eval/collect_benchmark_results.py \
    --logdir logs/ --resultdir results/ --scratchdir work-dir/

python scripts/plots/plot_benchmark_comparison.py \
    --sailir-csv results/benchmark_mixed_summary_v13.csv \
    --kira-csv   results/kira_benchmark.csv \
    --out        results/benchmark_comparison.pdf
```

`results/sailir_benchmark.csv` is the CSV used in the paper; pass it to
`--sailir-csv` to regenerate Fig. 3 verbatim.

### 6. Kira benchmark (optional)

```bash
export KIRA=/path/to/kira/bin/kira
export FERMATPATH=/path/to/fermat/fer64
cd scripts/kira/ && ./run_benchmark_s7.sh
```

## Pentagon-box (TA family)

The pentagon-box pipeline mirrors the trianglebox flow with the
`topology_input/pentagonbox` directory and the dedicated launcher set:

### Data-gen

```bash
export SAILIR_DIR=$(pwd)
export PYTHON=$(which python)

condor_submit data-gen/datagen_pentagonbox.jdl     # 100 × 1000 (= 1×)
# or:
condor_submit data-gen/datagen_pentagonbox_10x.jdl # 1000 × 1000 (= 10×)
```

Output JSONLs land in `data/pentagonbox_raw_jsonl/` (or
`data/pentagonbox_10x_raw_jsonl/` for the 10× run; seeds are offset to keep
them disjoint from the 1× set).

### Sharded preprocess (large datasets)

For the 10× dataset, packing all ~13M samples in a single Python process
would exhaust memory. Each worker JSONL is preprocessed independently into
its own packed shard:

```bash
bash data-gen/submit_preprocess_pentagonbox_10x_batched.sh
```

This submits the 1000 shards in throttled batches via Condor file transfer
(input staged into `$_CONDOR_SCRATCH_DIR`, output transferred back to
`data/pentagonbox_10x_packed/shard_<N>/`). Each shard packs to
`train.pt + val.pt + test.pt` (~115 MB total). At train time, wrap the
shards in `torch.utils.data.ConcatDataset` with `torch.load(..., mmap=True)`
to keep peak RAM bounded.

### Train

```bash
condor_submit training/train_pentagonbox.sub
```

Or directly:

```bash
python training/train_classifier.py \
    --topology   topology_input/pentagonbox \
    --data_dir   data/pentagonbox_packed/ \
    --output_dir checkpoints/pentagonbox/ \
    --epochs 30 --batch_size 128 --lr 4e-4 --prime 1009 --device cuda
```

### Reduce

Same as trianglebox, but with `--topology topology_input/pentagonbox` and an
11-index integral, e.g.:

```bash
python scripts/eval/onestep_worker.py \
    --topology topology_input/pentagonbox \
    --integral=1,0,1,0,1,1,-1,0,0,0,-1 \
    --model-checkpoint checkpoints/pentagonbox/best_model.pt \
    --output reduction.pkl \
    --beam_width 20 --prime 1009 -v
```

## Adding a new topology

See `topology_input/HOW_TO_DERIVE_FROM_KIRA.md`. In brief, run Kira to
extract: integralfamilies.yaml, kinematics.yaml, IBP+LI identity templates,
and the master basis. Drop them into `topology_input/<family>/` and every
script accepts that path via `--topology`.

## Citation

```bibtex
@article{Shih:2026jfe,
    author = "Shih, David",
    title = "{Learning to Unscramble Feynman Loop Integrals with SAILIR}",
    eprint = "2604.05034",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "4",
    year = "2026"
}
```

## License

MIT.
