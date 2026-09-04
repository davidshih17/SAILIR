"""
Sharded IterableDataset for SAILIR phase-2 pentagonbox training data.

Per-shard files are produced by
  scripts/data_gen/submit_preprocess_pentagonbox_10x_batched.sh
and live at
  <shards_dir>/shard_<N>/{train,val,test}.pt

Each shard's train.pt unpickles to ~420 MB (mostly the `subs_raw` Python list,
which mmap cannot help with). Holding all 1000 shards in RAM (~420 GB) is
infeasible on this box, so this loader:

  1. Per epoch, shuffles the shard order.
  2. Maintains a sliding pool of K=4 loaded shards (~1.7 GB working set).
  3. Yields one sample at a time, drawn uniformly at random over the pool's
     remaining unseen samples. When a shard drains, evict it and load the next
     queued shard.
  4. Splits the shard list across DataLoader workers so each worker reads a
     disjoint subset (no double loading).

Reuses PackedDatasetV5 from train_classifier.py for per-sample indexing into
the packed tensors, so the collate_fn output stays byte-identical to the
non-sharded path.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from torch.utils.data import IterableDataset, get_worker_info


# Import lives here at module top so workers don't re-parse train_classifier.py
# on every shard. PackedDatasetV5 is just a thin __getitem__ wrapper around the
# packed dict — no heavy state.
import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_classifier import PackedDatasetV5  # noqa: E402


def discover_shards(shards_dir: str | os.PathLike, split: str) -> List[Path]:
    """Return sorted list of <shards_dir>/shard_*/<split>.pt that actually exist."""
    root = Path(shards_dir)
    candidates = sorted(
        root.glob('shard_*'),
        key=lambda p: int(p.name.split('_')[-1]),
    )
    files = [d / f'{split}.pt' for d in candidates if (d / f'{split}.pt').is_file()]
    return files


def _count_shard_samples(path: str) -> tuple:
    """Worker function for parallel manifest build. Top-level so it pickles."""
    import torch as _torch
    d = _torch.load(path, weights_only=False)
    n = int(len(d['labels']))
    del d
    return (path, n)


def build_manifest(shards_dir: str | os.PathLike, splits=('train', 'val', 'test'),
                   force: bool = False, n_workers: int = 16) -> dict:
    """
    Cache per-shard sample counts in <shards_dir>/manifest.json so the training
    loop can compute exact per-rank batch counts without re-loading every shard.

    Returns the manifest dict: {'splits': {split: {shard_path: n}, ...},
    'totals': {split: n_total}}. Parallelized with `n_workers` processes.

    Cost: ~2-3 min for 2000 shards at n_workers=16. Cached to JSON; subsequent
    calls return instantly from disk unless force=True.
    """
    import time as _time
    from multiprocessing import Pool

    root = Path(shards_dir)
    manifest_path = root / 'manifest.json'
    if manifest_path.is_file() and not force:
        with open(manifest_path) as f:
            return json.load(f)

    manifest = {'splits': {}, 'totals': {}}
    for split in splits:
        files = discover_shards(root, split)
        if not files:
            manifest['splits'][split] = {}
            manifest['totals'][split] = 0
            continue
        t0 = _time.time()
        print(f"[manifest] scanning {len(files)} {split} shards with {n_workers} workers...", flush=True)
        # Pool.imap_unordered for progress feedback; chunksize amortizes IPC.
        chunksize = max(1, len(files) // (n_workers * 4))
        counts: dict = {}
        with Pool(n_workers) as pool:
            for i, (p, n) in enumerate(pool.imap_unordered(_count_shard_samples,
                                                           [str(f) for f in files],
                                                           chunksize=chunksize)):
                counts[p] = n
                if (i + 1) % 200 == 0 or (i + 1) == len(files):
                    elapsed = _time.time() - t0
                    rate = (i + 1) / elapsed
                    print(f"[manifest]   {split}: {i+1}/{len(files)}  "
                          f"({rate:.1f} shard/s, ETA {(len(files)-i-1)/rate:.0f}s)", flush=True)
        manifest['splits'][split] = counts
        manifest['totals'][split] = sum(counts.values())
        print(f"[manifest]   {split}: total {manifest['totals'][split]:,} samples "
              f"in {_time.time()-t0:.1f}s", flush=True)

    # Atomic write so a crash mid-build can't leave a corrupt manifest.
    tmp = manifest_path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)
    print(f"[manifest] wrote {manifest_path}", flush=True)
    return manifest


class ShardedIBPDataset(IterableDataset):
    """
    Buffered-shuffle IterableDataset over per-shard packed-tensor files.

    Args:
        shard_files: explicit list of shard .pt paths to iterate.
        shuffle: shuffle shard order per epoch and sample order within each
            shard. Required True for train, False for val/test.
        buffer_shards: K — number of shards held in memory simultaneously.
            Samples are drawn uniformly at random over the K shards' remaining
            unseen samples. K=1 reduces to per-shard shuffle; K>=2 mixes across
            shards. Default 4.
        seed: base RNG seed. The effective seed for each (epoch, worker) is
            seed + epoch * 1_000_003 + worker_id. Call set_epoch() each epoch.
    """

    def __init__(
        self,
        shard_files: Sequence[Path],
        shuffle: bool = True,
        buffer_shards: int = 4,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        token_budget: int = 0,
        max_batch: int = 0,
        sort_window: int = 2048,
    ):
        super().__init__()
        # TOKEN BUDGET (added 2026-08-10). When > 0 the iterator yields LISTS of
        # samples obeying len(batch) * max_actions_in_batch <= token_budget,
        # instead of one sample at a time. The caller then uses
        # DataLoader(batch_size=None), so the list goes straight to collate_fn.
        #
        # Why this is needed: the --data_dir path bounds padded-batch GPU memory
        # with TokenBudgetBatchSampler, but an IterableDataset cannot take a
        # batch_sampler, so the shards path fell back to a flat batch_size=64.
        # With ~12,700 actions per sample that is ~813k tokens, 12.4x over the
        # 65,536 budget -- it OOM'd a 15.6 GB GPU inside the action encoder
        # (tried to allocate 6.66 GiB in one torch.cat). It only ever worked on
        # 40-80 GB cards. token_budget=0 keeps the old per-sample behaviour.
        self.token_budget = int(token_budget)
        self.max_batch = int(max_batch) if max_batch else 10**9
        self.sort_window = max(1, int(sort_window))
        self.shard_files = [Path(p) for p in shard_files]
        self.shuffle = shuffle
        self.buffer_shards = int(buffer_shards) if shuffle else 1
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch before iterating, so the per-epoch shuffle is reproducible."""
        self._epoch = int(epoch)

    def __iter__(self):
        # Two-level partition: first by rank (DDP), then by DataLoader worker.
        # Each rank sees a disjoint set of shards; within a rank each worker
        # takes a further disjoint stride.
        files_for_rank = list(self.shard_files[self.rank::self.world_size])

        worker_info = get_worker_info()
        if worker_info is None:
            files = files_for_rank
            worker_id = 0
        else:
            files = files_for_rank[worker_info.id::worker_info.num_workers]
            worker_id = worker_info.id

        rng = random.Random(
            self.seed + self._epoch * 1_000_003 + self.rank * 1_000_037 + worker_id
        )
        if self.shuffle:
            rng.shuffle(files)

        # in-progress token-budget batch (unused when token_budget <= 0)
        cur: list = []
        cur_max = 0

        # Active pool: each entry is [PackedDatasetV5, list_of_remaining_indices].
        pool: List[list] = []
        queue: List[Path] = list(files)

        def _load_one() -> bool:
            if not queue:
                return False
            path = queue.pop(0)
            d = torch.load(path, weights_only=False)
            n = len(d['labels'])
            idxs = list(range(n))
            if self.shuffle:
                rng.shuffle(idxs)
            pool.append([PackedDatasetV5(d), idxs])
            return True

        # Initial fill
        while len(pool) < self.buffer_shards and _load_one():
            pass

        while pool:
            # Pick a shard weighted by its remaining sample count → uniform
            # over the pool's combined remaining samples.
            sizes = [len(entry[1]) for entry in pool]
            total = sum(sizes)
            if total == 0:
                break
            r = rng.randrange(total)
            cum = 0
            picked = 0
            for picked, sz in enumerate(sizes):
                cum += sz
                if r < cum:
                    break
            ds, idxs = pool[picked]
            # idxs was pre-shuffled, so pop() gives a random remaining sample.
            sample_idx = idxs.pop()
            s = ds[sample_idx]

            if self.token_budget <= 0:
                yield s
            else:
                # Collect a WINDOW, then sort it by action count before
                # batching. Greedy batching of a randomly-ordered stream packs
                # terribly: one large sample either rides alone or truncates the
                # batch being accumulated, and measured on this corpus that gave
                # a MEDIAN BATCH SIZE OF 1 (23.5k batches into an epoch that
                # should be a few thousand). Sorting groups similar sizes so the
                # budget buys many small samples per batch and only genuinely
                # huge ones ride alone -- which is what TokenBudgetBatchSampler
                # achieves globally on the --data_dir path.
                # Randomness is preserved at window granularity: the window is
                # drawn at random from the shard pool, only its internal order
                # is sorted, and the emitted batches are shuffled before use.
                cur.append(s)
                if len(cur) >= self.sort_window:
                    for b in self._emit(cur, rng):
                        yield b
                    cur = []

            if not idxs:
                # Shard exhausted — evict and pull in the next one.
                del pool[picked]
                _load_one()

        # Flush the tail window. A partial window is still real data; dropping
        # it would silently discard samples every epoch.
        if self.token_budget > 0 and cur:
            for b in self._emit(cur, rng):
                yield b

    def _emit(self, window, rng):
        """Sort a window by action count and cut it into budgeted batches.

        Same invariant as TokenBudgetBatchSampler: len(batch) * max_actions in
        that batch <= token_budget, and len(batch) <= max_batch. Sorting is what
        makes the budget buy a useful number of samples -- unsorted, one large
        sample caps the whole batch.
        """
        window.sort(key=lambda s: len(s['action_ibp_ops']))
        batches, cur, cur_max = [], [], 0
        for s in window:
            n = len(s['action_ibp_ops'])
            m = max(cur_max, n)
            if cur and ((len(cur) + 1) * m > self.token_budget
                        or len(cur) >= self.max_batch):
                batches.append(cur)
                cur, cur_max = [s], n
            else:
                cur.append(s)
                cur_max = m
        if cur:
            batches.append(cur)
        # Sorting made batches size-ordered; shuffle so the model does not see
        # every epoch's batches in ascending difficulty.
        rng.shuffle(batches)
        return batches


def estimate_samples_per_epoch(
    shard_files: Sequence[Path],
    manifest: Optional[dict] = None,
    split: str = 'train',
    sample_n: int = 5,
) -> int:
    """
    Total sample count across `shard_files`.

    If a manifest is provided, sums exact counts. Otherwise loads `sample_n`
    shards, computes the mean, multiplies by len(shard_files) for an estimate.
    """
    if manifest is not None and split in manifest.get('splits', {}):
        per_file = manifest['splits'][split]
        keys = set(str(p) for p in shard_files)
        known = [per_file[k] for k in per_file if k in keys]
        if len(known) == len(shard_files):
            return sum(known)
        # Fall through to estimate if manifest is incomplete.
    if not shard_files:
        return 0
    pick = list(shard_files)[: sample_n]
    counts = []
    for p in pick:
        d = torch.load(p, weights_only=False)
        counts.append(len(d['labels']))
        del d
    mean = sum(counts) / len(counts)
    return int(round(mean * len(shard_files)))
