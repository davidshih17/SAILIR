#!/usr/bin/env python
"""Pack ONE shard of recordings. One Condor job per shard.

Replaces the single-process build, which was the worst bottleneck in the loop:
~2 hours wall and ~84 GB RSS for one corpus, while the RECORDING that produced
it ran 1,870-wide on the farm. Nothing about packing is inherently serial --
each recording is independent -- so this splits the file list and lets the farm
do it, writing the <shards_dir>/shard_K/{train,val}.pt layout the trainer
already discovers via --shards_dir.

Expected: ~1,820 files over 32 shards = ~57 files/job, a few minutes and a few
GB each, versus 2 hours and 84 GB in one process.

TWO DELIBERATE SEMANTIC CHANGES, both consequences of sharding:

1. THE SPLIT IS BY FILE (trajectory), NOT BY SAMPLE. The old build shuffled all
   samples globally and took the first 10% as val, which cut trajectories in
   half -- measured previously, 756 of 890 trajectories in one shard's train.pt
   were missing steps that had landed in val/test. That is also leakage: steps
   from the same reduction appear on both sides. Splitting whole files fixes
   both, and is the only split that can be decided locally by a shard job with
   no global state.

2. ASSIGNMENT IS BY crc32 OF THE FILENAME, not Python's hash(), which is salted
   per process and would give a different split in every job -- silently
   producing overlapping or missing shards.

The two changes mean a shard-packed corpus is NOT bit-comparable with an
existing single-process pack. Use it for new corpora, or repack an old one
whole; do not mix.

NO REPLAY (decided 2026-08-10). The single-process builds mixed in ~61,725
samples from the original scramble corpus, but they were never trained on --
training filtered to `scramble_id != -1` -- and only the ~10% that happened to
fall in the val slice were kept, as an `--extra_val` set reporting whether
fine-tuning was degrading the base distribution. Two reasons it is gone:
the numbers were never comparable across arms (a random 10% slice that moves
with the split seed), and every arm fine-tunes from the same init.pt rather
than sequentially, so there is no accumulating drift for the tripwire to catch.
Dropping it is what makes this packer a COMPLETE replacement for the old build
rather than one needing an extra replay shard.
"""
import argparse
import glob
import hashlib
import os
import sys
import zlib
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent))

import torch
from sailir.topology import Topology
from preprocess_to_tensors import load_and_convert, pack_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topology', required=True)
    ap.add_argument('--recdir', required=True,
                    help='directory of rec_*.jsonl recordings')
    ap.add_argument('--outdir', required=True,
                    help='shards_dir; this job writes <outdir>/shard_<shard>/')
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--val-every', type=int, default=10,
                    help='1 in N FILES goes to val (default 10 = 10%%)')
    ap.add_argument('--keep-subs', action='store_true',
                    help='ALSO pack the substitution fields. Off by default: '
                         'the nosubs model deletes every sub_* tensor on the '
                         'first line of forward(), yet they are 44.6%% of the '
                         'packed bytes (subs_raw alone 195 MB of a 448 MB '
                         'shard) and the sharded loader re-reads and '
                         're-unpickles them EVERY epoch. Pass this only to '
                         'build a corpus for the full IBPActionClassifier.')
    args = ap.parse_args()

    topology = Topology.from_dir(args.topology)
    files = sorted(f for f in glob.glob(os.path.join(args.recdir, 'rec_*.jsonl'))
                   if os.path.getsize(f) > 0)
    if not files:
        raise SystemExit(f'no non-empty recordings under {args.recdir}')

    def _h(p, salt=b''):
        # RECOVERY FILES CO-HASH WITH THEIR SOURCE. rec_<tag>__recovery.jsonl
        # holds scramble-corrupted states derived from rec_<tag>.jsonl, so it
        # must land in the SAME shard and the SAME train/val side. Otherwise a
        # one-identity perturbation of a val state sits in train -- near
        # duplicate leakage, on exactly the metric used to judge whether the
        # recovery data helped.
        b = os.path.basename(p).replace('__recovery', '')
        return zlib.crc32(salt + b.encode()) & 0xffffffff

    # A DIFFERENT HASH FAMILY for the val decision, not a salted crc32.
    #
    # Using one crc32 for both couples them: values with crc32 % nshards == k
    # form an arithmetic progression, so crc32 % val_every reaches only
    # gcd-many residues -- with nshards=32 and val_every=10, only EVEN shards
    # could hold a val file (16 of 32, observed twice).
    #
    # Salting the crc32 does NOT fix this, which is the non-obvious part:
    # CRC32 is linear over GF(2), so prepending a fixed prefix maps the digest
    # through a fixed linear transform, and that transform preserves parity.
    # Measured: with b'val:' the val files still landed in even shards only.
    # SHA1 is not linear, and measured on this corpus it spreads val files over
    # 32/32 shards at 10.8%.
    #
    # The val FRACTION was always correct either way (membership is per file),
    # so this is about even distribution, not correctness.
    def _is_val(p):
        return int(hashlib.sha1(os.path.basename(p).encode()).hexdigest()[:8],
                   16) % args.val_every == 0

    mine = [f for f in files if _h(f) % args.nshards == args.shard]
    tr = [f for f in mine if not _is_val(f)]
    va = [f for f in mine if _is_val(f)]
    print(f'shard {args.shard}/{args.nshards}: {len(mine)} of {len(files)} '
          f'files (train {len(tr)}, val {len(va)})', flush=True)

    out = Path(args.outdir) / f'shard_{args.shard}'
    out.mkdir(parents=True, exist_ok=True)

    for name, sel in (('train', tr), ('val', va)):
        if not sel:
            # A shard may legitimately get no val files; write nothing rather
            # than an empty tensor file, since discover_shards() skips missing
            # splits but would happily load a zero-row one into the sampler.
            print(f'  {name}: no files, skipping', flush=True)
            continue
        samples = load_and_convert(sel, topology.n_indices)
        n_c = sum(len(s.get('valid_label_idxs') or []) for s in samples)
        packed = pack_samples(samples, include_subs=args.keep_subs)
        torch.save(packed, out / f'{name}.pt')
        mb = (out / f'{name}.pt').stat().st_size / 1e6
        print(f'  {name}: {len(samples)} samples, {mb:.1f} MB, '
              f'mean candidates/step {n_c/max(len(samples),1):.1f}', flush=True)
        del samples, packed

    print(f'SHARD {args.shard} DONE', flush=True)


if __name__ == '__main__':
    main()
