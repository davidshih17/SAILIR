#!/usr/bin/env python
"""Load a packed recipe index, verify regeneration, measure hit rate.

Three questions, in order of how much they matter:
  1. Does a packed recipe regenerate the SAME legal reduction? (correctness)
  2. What fraction of the live frontier does the index already cover? (top-down value)
  3. How much of the index is below the frontier? (bottom-up value)
"""
import os, pickle, sys, time
import numpy as np

R = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, R); sys.path.insert(0, R + '/reduction')
from sailir.topology import Topology
from sailir import ibp_env as ie
import topo_config as tc
ie.init_from_topology(Topology.from_dir(tc.TOPO_DIR))
ie.set_prime(int(os.environ.get('SAILIR_PRIME', '101')))
ie.set_paper_masters_only(False)
from canonical_masters import apply_canonical_masters; apply_canonical_masters()
from sailir.ibp_env import is_master, IBPEnvironment
from total_order import tkey
from build_recipe_index import pack, unpack, N_IDX


class RecipeIndex:
    """Sorted packed keys + binary search. No Python dict, no per-entry object."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.keys = z['keys']; self.vals = z['vals']
        # one uint8 row -> one comparable scalar via void view, so np.searchsorted
        # orders rows lexicographically without materialising tuples
        self._flat = np.ascontiguousarray(self.keys).view(
            np.dtype((np.void, N_IDX))).ravel()

    def __len__(self):
        return len(self.keys)

    def get(self, integral):
        k = np.frombuffer(pack(integral), dtype=np.uint8).view(
            np.dtype((np.void, N_IDX)))[0]
        i = np.searchsorted(self._flat, k)
        if i >= len(self._flat) or self._flat[i] != k:
            return None
        v = self.vals[i]
        return int(v[0]), unpack(bytes(v[1:]))


def main():
    idx = RecipeIndex(sys.argv[1] if len(sys.argv) > 1
                      else R + '/results/recipe_index_sample.npz')
    print(f"  index: {len(idx):,} recipes", flush=True)
    env = IBPEnvironment()

    # ---- 1. regeneration is faithful and legal -------------------------------
    rng = np.random.default_rng(0)
    probe = rng.choice(len(idx), size=min(500, len(idx)), replace=False)
    ok = bad = 0; t0 = time.time()
    for i in probe:
        X = unpack(bytes(idx.keys[i]))
        op, seed = idx.get(X)
        raw = env.get_raw_equation_cached(op, seed)
        if not raw or min(raw, key=tkey) != X:
            bad += 1; continue
        if all(tkey(k) > tkey(X) or is_master(k) for k in raw if k != X):
            ok += 1
        else:
            bad += 1
    dt = (time.time() - t0) / len(probe) * 1e3
    print(f"  regenerate: {ok}/{len(probe)} legal and identical, {bad} bad, "
          f"{dt:.3f} ms/lookup+regen", flush=True)

    # ---- 2 & 3. against the live orchestrator state --------------------------
    st = sys.argv[2] if len(sys.argv) > 2 else None
    if not st or not os.path.exists(st):
        print("  (no orchestrator state given -- skipping hit rate)"); return
    with open(st, 'rb') as f:
        S = pickle.load(f)
    cache = S.get('cache') or {}
    frontier = [tuple(k) for k in S.get('non_masters', []) or []]
    if not frontier:
        frontier = [tuple(k) for k in cache.get('__frontier__', [])] if cache else []
    print(f"  state: cache {len(cache):,}   frontier {len(frontier):,}", flush=True)

    have = sum(1 for k in frontier if k in cache)
    hit = sum(1 for k in frontier if k not in cache and idx.get(k) is not None)
    need = len(frontier) - have
    print(f"  frontier already in cache : {have:,}", flush=True)
    print(f"  frontier NOT in cache     : {need:,}", flush=True)
    if need:
        print(f"  ... of which the INDEX covers: {hit:,}  ({100.0*hit/need:.1f}%)",
              flush=True)

    # how much of the index is strictly BELOW the frontier -- i.e. only reachable
    # by a bottom-up sweep, useless to a top-down one
    if frontier:
        worst = max(tkey(k) for k in frontier)   # largest tkey = lowest integral
        below = sum(1 for i in range(0, len(idx), max(1, len(idx)//20000))
                    if tkey(unpack(bytes(idx.keys[i]))) > worst)
        samp = len(range(0, len(idx), max(1, len(idx)//20000)))
        print(f"  index entries BELOW the whole frontier: ~{100.0*below/samp:.1f}% "
              f"(sampled {samp:,})", flush=True)


if __name__ == '__main__':
    main()
