#!/usr/bin/env python
"""RECIPE INDEX: {reducible integral -> (op, seed)}, packed to 31 bytes.

WHY A RECIPE, NOT A REDUCTION. Every step of a finished worker used an exact IBP
identity raw = get_raw_equation_cached(op, seed) with seed = target + delta.
Solving that identity for its MAXIMAL element X is a valid one-level reduction
of X. Storing the EXPANSION costs ~4 KB resident each and mining yields ~9.1x
more integrals than the campaign solves -- that is Laporta's reduction table
rebuilt. Storing the RECIPE costs 31 B measured, and only a hit is ever
materialised.

  key   = 15 bytes  (X, each index +128 so it fits a uint8)
  value =  1 + 15   (op, then the seed)

EFFICIENCY -- the three things that dominate, and what is done about them.

1. tkey is the hot cost (it sums, builds an abs tuple, and does a sector-rank
   lookup: ~4 passes over 15 ints). The naive form calls it 3n times per
   identity -- n for the min, then tkey(k) and a RECOMPUTED tkey(X) for each
   legality test. Here it is called exactly n times, once per term.

2. The legality test is free. X is the ARGMIN of tkey, so tkey(k) >= tkey(X)
   always, and "tkey(k) > tkey(X)" can fail ONLY on an exact tie. So instead of
   n comparisons the check is: every other term whose tkey ties the minimum must
   be a master. Ties are rare, so this is a branch that almost never fires.
   Mathematically identical to the original all(...) form.

3. No dict. At ~20M entries a {bytes: bytes} dict is ~3 GB of pure per-object
   overhead. Records stream into a bytearray and out to a flat binary file at
   O(1) memory; dedup happens once at merge, in C, via argsort over a void view.
   Duplicates cost 31 bytes of sequential I/O instead of a hash insert.

Sharding: --shard i --nshards N partitions the file list so the build can run in
parallel; --merge combines the shards and writes the final .npz.

STAMPED with topology and prime -- regeneration is a correctness dependency, so
an index built against different IBP tables must never be silently reused.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

R = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, R); sys.path.insert(0, R + '/reduction')

N_IDX = 15
OFF = 128
REC_OUT = 2 * N_IDX + 1      # 31 bytes: key(15) + op(1) + seed(15)
REC = REC_OUT + 2            # 33 bytes on the SHARD: + 2 quality bytes
FLUSH = 1 << 22              # 4 MB of records between writes

# The same integral X is reducible from many different identities -- 46% of
# mined records are duplicate keys. They are all legal, but not equally good:
# each expands X into a different set of children. Two free quality bytes let
# the merge keep the BEST expansion instead of whichever file sorted first.
#
#   byte 0  SHALLOW: children in the same sector as X (tkey[0] equal). These
#           are the ones that barely descend -- the pathology seen in a real
#           496-step walk, where 11 of 62 children shared the target's exact
#           (props, r, s) and differed only in the |abs| tiebreak.
#   byte 1  WIDTH: total children.
#
# Ordered SHALLOW first, then WIDTH. Both come from tkeys already computed, so
# they cost nothing, and they are dropped after the merge picks -- the final
# index stays 31 B/entry.


def _results_dir():
    return R + '/results/gr_reduce/g1023_greedy_certified/work/results'


def _patch_eval_coeff(ie):
    """2.60x on get_raw_equation, verified bit-identical on 33,733 equations
    (368,288 coefficient terms). Three redundancies in the original, all per
    TERM even though `seed` is constant across a template:

      1. eval() on a STRING re-parses the expression every call.
      2. coeff_str.replace('^','**') redone every call.
      3. ns = {f'a{i}': seed[i] ...} rebuilt every call -- 15 f-string formats
         per term, ~15 terms per equation, so 15x redundant.

    Fixed by compiling each coefficient once (the template set is finite, so
    the cache is bounded) and memoising the namespace for the current seed.

    Applied as a monkey-patch: sailir/ibp_env.py is shared with the live
    campaign's workers, so it is not edited from here.
    """
    orig = ie.eval_coeff
    code_cache = {}
    last = [None, None]

    def eval_coeff(coeff_str, seed):
        ent = code_cache.get(coeff_str)
        if ent is None:
            ent = (compile(coeff_str.replace('^', '**'), '<c>', 'eval'),
                   '/' in coeff_str)
            code_cache[coeff_str] = ent
        code, has_div = ent
        if last[0] is not seed:
            ns = {f'a{i}': seed[i] for i in range(ie.N_INDICES)}
            ns.update(ie.KINEMATICS)
            last[0] = seed
            last[1] = ns
        ns = last[1]
        try:
            if has_div:
                from fractions import Fraction
                fns = {k: Fraction(v) for k, v in ns.items()}
                fr = Fraction(eval(code, {"__builtins__": {}}, fns))
                return (fr.numerator
                        * pow(fr.denominator % ie.PRIME, ie.PRIME - 2, ie.PRIME)
                        % ie.PRIME)
            return eval(code, {"__builtins__": {}}, ns) % ie.PRIME
        except Exception:
            return 0

    ie.eval_coeff = eval_coeff
    return orig


def build(shard, nshards, outpath, limit):
    from sailir.topology import Topology
    from sailir import ibp_env as ie
    import topo_config as tc
    ie.init_from_topology(Topology.from_dir(tc.TOPO_DIR))
    prime = int(os.environ.get('SAILIR_PRIME', '101'))
    ie.set_prime(prime)
    ie.set_paper_masters_only(False)
    from canonical_masters import apply_canonical_masters
    apply_canonical_masters()
    from sailir.ibp_env import is_master, IBPEnvironment
    from total_order import tkey
    if os.environ.get('SAILIR_FAST_COEFF', '1') == '1':
        _patch_eval_coeff(ie)

    D = _results_dir()
    names = sorted(n for n in os.listdir(D) if n.endswith('.pkl'))
    if nshards > 1:
        names = names[shard::nshards]
    if limit:
        names = names[:limit]
    env = IBPEnvironment()
    raw_eq = env.get_raw_equation_cached          # bind once, called per step

    buf = bytearray()
    rec = [0] * REC                              # reused; never reallocated
    n_w = n_steps = n_kept = n_illegal = n_range = 0
    t0 = time.time()
    join = os.path.join
    with open(outpath, 'wb') as fout:
        write = fout.write
        for fi, n in enumerate(names):
            try:
                with open(join(D, n), 'rb') as f:
                    d = pickle.load(f)
            except Exception:
                continue
            if not d.get('success'):
                continue
            path = d.get('path')
            if not path:
                continue                         # steps=0 symmetry route
            n_w += 1
            for tgt, op, delta in path:
                if not (0 <= op < 256):
                    n_range += 1
                    continue
                seed = [tgt[i] + delta[i] for i in range(N_IDX)]
                try:
                    raw = raw_eq(op, tuple(seed))
                except Exception:
                    continue
                if not raw or len(raw) < 2:
                    continue
                n_steps += 1

                # ONE tkey per term, collected once so the quality bytes are
                # free. ~11 terms per identity, so the list is tiny.
                keyed = [(tkey(k), k) for k in raw]
                best_t, best_k = keyed[0]
                for t, k in keyed:
                    if t < best_t:
                        best_t = t; best_k = k

                # Legality: X is the argmin, so every other term already has a
                # strictly larger tkey EXCEPT exact ties. Only ties need testing.
                bad = False
                sector0 = best_t[0]
                shallow = 0
                for t, k in keyed:
                    if k is best_k:
                        continue
                    if t == best_t and not is_master(k):
                        bad = True
                        break
                    if t[0] == sector0:
                        shallow += 1
                if bad:
                    n_illegal += 1
                    continue
                width = len(keyed) - 1
                if shallow > 255:
                    shallow = 255
                if width > 255:
                    width = 255

                j = 0
                for v in best_k:
                    v += OFF
                    if v < 0 or v > 255:
                        break
                    rec[j] = v; j += 1
                else:
                    rec[N_IDX] = op
                    j = N_IDX + 1
                    for v in seed:
                        v += OFF
                        if v < 0 or v > 255:
                            break
                        rec[j] = v; j += 1
                    else:
                        rec[REC_OUT] = shallow
                        rec[REC_OUT + 1] = width
                        buf.extend(rec)
                        n_kept += 1
                        if len(buf) >= FLUSH:
                            write(buf); buf.clear()
                        continue
                n_range += 1
            if (fi + 1) % 25000 == 0:
                print(f"    {fi+1:7,}/{len(names):,} files  {n_kept:11,} records  "
                      f"{time.time()-t0:5.0f}s", flush=True)
        if buf:
            write(buf)
    print(f"  shard {shard}/{nshards}: workers {n_w:,}  steps {n_steps:,}  "
          f"records {n_kept:,}  illegal {n_illegal:,}  out-of-range {n_range:,}  "
          f"{time.time()-t0:.0f}s", flush=True)
    print(f"  wrote {outpath} ({os.path.getsize(outpath)/1e6:.1f} MB)", flush=True)


def merge(shards, outpath):
    """Sort + dedupe by key in C. Keeps the first recipe seen for each integral."""
    t0 = time.time()
    parts = []
    for p in shards:
        a = np.fromfile(p, dtype=np.uint8)
        if a.size:
            parts.append(a.reshape(-1, REC))
        print(f"    {p}: {a.size // REC:,} records", flush=True)
    if not parts:
        print("  nothing to merge"); return
    a = parts[0] if len(parts) == 1 else np.concatenate(parts)
    del parts
    n_raw = len(a)
    # Dedupe by the 15-byte key. Pad to 16 and view as 2x uint64: integer
    # compares have real ufunc loops (void does not), and unlike an 'S15' view
    # they cannot silently equate keys that differ only in trailing zero bytes.
    k16 = np.zeros((n_raw, 16), dtype=np.uint8)
    k16[:, :N_IDX] = a[:, :N_IDX]
    # BIG-endian view: integer order then equals BYTE-LEXICOGRAPHIC order, which
    # is what RecipeIndex's searchsorted over a void view compares. A native
    # (little-endian) uint64 view sorts by byte-reversed value, leaving the index
    # sorted one way and searched another -- every lookup misses, silently.
    u = k16.view('>u8').reshape(-1, 2)
    # lexsort's LAST key is primary: group by key, and within a group order by
    # SHALLOW then WIDTH so the first row of each group is the best expansion.
    order = np.lexsort((a[:, REC_OUT + 1], a[:, REC_OUT],
                        u[:, 1], u[:, 0]))
    us = u[order]
    del k16, u
    first = np.empty(n_raw, dtype=bool)
    first[0] = True
    np.logical_or(us[1:, 0] != us[:-1, 0], us[1:, 1] != us[:-1, 1],
                  out=first[1:])
    del us
    sel = order[first]
    del order, first
    out = a[sel]
    kept_shallow = float(out[:, REC_OUT].mean()) if len(out) else 0.0
    kept_width = float(out[:, REC_OUT + 1].mean()) if len(out) else 0.0
    all_shallow = float(a[:, REC_OUT].mean()) if n_raw else 0.0
    all_width = float(a[:, REC_OUT + 1].mean()) if n_raw else 0.0
    del a, sel
    np.savez_compressed(
        outpath, keys=np.ascontiguousarray(out[:, :N_IDX]),
        vals=np.ascontiguousarray(out[:, N_IDX:REC_OUT]),
        meta=np.array([os.environ.get('SAILIR_TOPOLOGY', '?'),
                       os.environ.get('SAILIR_PRIME', '101'),
                       str(n_raw)], dtype=object))
    nb = len(out) * REC_OUT
    print(f"  {n_raw:,} records -> {len(out):,} distinct "
          f"({100.0*len(out)/max(n_raw,1):.1f}% unique)", flush=True)
    print(f"  wrote {outpath} ({os.path.getsize(outpath)/1e6:.1f} MB on disk, "
          f"{nb:,} B resident = {nb/max(len(out),1):.0f} B/entry, "
          f"{time.time()-t0:.0f}s)", flush=True)
    print(f"  quality: shallow-children {all_shallow:.2f} over ALL records -> "
          f"{kept_shallow:.2f} for the KEPT recipe; "
          f"width {all_width:.2f} -> {kept_width:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--merge', nargs='*', default=None,
                    help='shard .bin files to merge into `out`')
    a = ap.parse_args()
    if a.merge is not None:
        merge(a.merge, a.out)
    else:
        build(a.shard, a.nshards, a.out, a.limit)


if __name__ == '__main__':
    main()
