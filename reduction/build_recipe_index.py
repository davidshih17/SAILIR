#!/usr/bin/env python
"""RECIPE INDEX: {reducible integral -> (op, seed)}, packed.

WHY A RECIPE, NOT A REDUCTION. Every step of a finished worker used an exact IBP
identity raw = get_raw_equation_cached(op, seed) with seed = target + delta
(greedy_reduce.py:647). Solving that identity for its MAXIMAL element X gives a
valid one-level reduction of X -- verified legal on 8,576/8,576 identities.
Storing the expansion costs ~390 B pickled / ~4 KB resident per entry, and the
mined set does not saturate (~127 new per worker at 800 workers), so
materialising it is a 40-50x blow-up -- which is just Laporta's reduction table
rebuilt opportunistically.

The recipe (op, seed) regenerates the same reduction deterministically in
0.028 ms, and packs into 16 bytes. Index of 58M entries: ~0.9 GB instead of
~23 GB. Nothing is materialised until the moment of use.

PACKING. Indices live in a small range, so each is one byte offset by +128.
  key   = 15 bytes  (the reducible integral X)
  value = 16 bytes  (op, then the 15-index seed)
Stored as two sorted uint8 arrays; lookup is a binary search, no Python dict.

STAMPED with topology and prime: regeneration is now a CORRECTNESS dependency,
so an index built against different IBP tables must not be silently reused.
"""
import os, pickle, random, sys, time
import numpy as np

R = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, R); sys.path.insert(0, R + '/reduction')
from sailir.topology import Topology
from sailir import ibp_env as ie
import topo_config as tc
ie.init_from_topology(Topology.from_dir(tc.TOPO_DIR))
PRIME = int(os.environ.get('SAILIR_PRIME', '101'))
ie.set_prime(PRIME)
ie.set_paper_masters_only(False)
from canonical_masters import apply_canonical_masters; apply_canonical_masters()
from sailir.ibp_env import is_master, IBPEnvironment
from total_order import tkey

N_IDX = 15
OFF = 128          # index range is small; +128 makes every component a uint8


def pack(t):
    return bytes(x + OFF for x in t)


def unpack(b):
    return tuple(x - OFF for x in b)


def main():
    D = R + '/results/gr_reduce/g1023_greedy_certified/work/results'
    out = sys.argv[1] if len(sys.argv) > 1 else R + '/results/recipe_index.npz'
    limit = int(os.environ.get('NW', '2000'))

    names = [n for n in os.listdir(D) if n.endswith('.pkl')]
    random.Random(0).shuffle(names)
    env = IBPEnvironment()

    recipes = {}          # packed X -> packed (op, seed)
    n_w = n_steps = n_illegal = n_range = 0
    t0 = time.time()
    for n in names:
        if n_w >= limit:
            break
        try:
            with open(os.path.join(D, n), 'rb') as f:
                d = pickle.load(f)
        except Exception:
            continue
        if not d.get('success'):
            continue
        path = d.get('path') or []
        if not path:
            continue            # symmetry-routed: steps=0, nothing to mine
        n_w += 1
        for (tgt, op, delta) in path:
            tgt = tuple(tgt); delta = tuple(delta)
            seed = tuple(tgt[i] + delta[i] for i in range(N_IDX))
            try:
                raw = env.get_raw_equation_cached(op, seed)
            except Exception:
                continue
            if not raw:
                continue
            n_steps += 1
            X = min(raw, key=tkey)                       # smallest tkey = highest
            others = [k for k in raw if k != X]
            if not others:
                continue
            # legal iff every other term is strictly lower, or terminal
            if not all(tkey(k) > tkey(X) or is_master(k) for k in others):
                n_illegal += 1
                continue
            if not all(-OFF < v < OFF for v in X) or not all(-OFF < v < OFF for v in seed):
                n_range += 1
                continue
            if not (0 <= op < 256):
                n_range += 1
                continue
            recipes.setdefault(pack(X), bytes([op]) + pack(seed))
        if n_w % 250 == 0:
            print(f"    {n_w:6d} workers  {len(recipes):9,} recipes  "
                  f"{time.time()-t0:5.0f}s", flush=True)

    print(f"  workers {n_w:,}  steps {n_steps:,}  illegal {n_illegal:,}  "
          f"out-of-range {n_range:,}", flush=True)
    print(f"  DISTINCT recipes: {len(recipes):,}", flush=True)

    keys = np.frombuffer(b''.join(sorted(recipes)), dtype=np.uint8).reshape(-1, N_IDX)
    order = sorted(recipes)
    vals = np.frombuffer(b''.join(recipes[k] for k in order),
                         dtype=np.uint8).reshape(-1, N_IDX + 1)
    np.savez_compressed(out, keys=keys, vals=vals,
                        meta=np.array([os.environ.get('SAILIR_TOPOLOGY', '?'),
                                       str(PRIME), str(n_w)], dtype=object))
    sz = os.path.getsize(out)
    print(f"  wrote {out}  ({sz/1e6:.1f} MB on disk, "
          f"{keys.nbytes+vals.nbytes:,} B in memory = "
          f"{(keys.nbytes+vals.nbytes)/max(len(recipes),1):.0f} B/entry)", flush=True)
    print(f"  projected at 20M entries: "
          f"{20e6*(keys.nbytes+vals.nbytes)/max(len(recipes),1)/1e9:.2f} GB resident", flush=True)


if __name__ == '__main__':
    main()
