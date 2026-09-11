#!/usr/bin/env python
"""Dispatch-time lookup of mined one-level reductions.

WHAT THIS IS FOR. A worker walk reducing target T uses an exact IBP identity at
every step. Solving each identity for its maximal element yields a valid
one-level reduction of that element -- free, already paid for. Measured on 386
walks / 25,177 identities: 25,034 (99.4%) of those elements sit strictly ABOVE
the walk's own target, 143 are the target itself, and 0 are below. So the mined
reductions are only ever cashable when the campaign dispatches BOTTOM-UP; a
top-down sweep has already retired them (holdout: 0 hits in 5,000 targets).

WHY AN INDEX AND NOT CACHE ENTRIES. Mining produces ~9.1x more integrals than
the campaign solves, and only 6.0% of them are ever dispatched as targets. Put
the expansions in `cache` and that is Laporta's reduction table rebuilt: ~4 KB
resident each, ~23 GB at 20M. Stored as the RECIPE (op, seed) it is 31 B/entry
measured -- 0.62 GB at 20M -- and the 94% that nobody asks for never costs more
than its 31 bytes. Only a HIT is ever materialised, one rule at a time.

TRUST NOTHING, VERIFY ON USE. Regeneration plus the legality check costs 0.2 ms.
That is cheap enough to re-derive the reduction every time instead of trusting
the stored recipe, which makes a stale index (different prime, different master
set, different topology) harmless: a recipe that no longer verifies is dropped
and the target dispatches to a worker as usual. An unverified rule fed into the
fold would trip the descent assertion and kill the run.
"""
import os

import numpy as np

N_IDX = 15
OFF = 128          # index components are small; +128 makes each a uint8


def pack(t):
    return bytes(int(x) + OFF for x in t)


def unpack(b):
    return tuple(x - OFF for x in b)


class RecipeIndex:
    """Sorted packed keys + binary search.

    Deliberately NOT a dict: 20M+ entries as Python tuples/bytes objects costs
    GB in per-object overhead alone. Two contiguous uint8 arrays cost exactly
    31 B/entry and searchsorted over a void view orders rows lexicographically
    without materialising anything per lookup.
    """

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.keys = np.ascontiguousarray(z['keys'])
        self.vals = np.ascontiguousarray(z['vals'])
        self.meta = z['meta'] if 'meta' in z else None
        self._flat = self.keys.view(np.dtype((np.void, N_IDX))).ravel()
        # searchsorted is only valid on a sorted array, and the failure mode is
        # SILENT: an index sorted by anything other than byte-lexicographic
        # order misses every lookup and reads as "mining is worthless". Check it
        # once at load -- one pass, and it turns that into a loud error.
        if len(self.keys) > 1:
            k16 = np.zeros((len(self.keys), 16), dtype=np.uint8)
            k16[:, :N_IDX] = self.keys
            v = k16.view('>u8').reshape(-1, 2)      # big-endian == byte order
            bad = int(np.count_nonzero(
                (v[1:, 0] < v[:-1, 0])
                | ((v[1:, 0] == v[:-1, 0]) & (v[1:, 1] < v[:-1, 1]))))
            if bad:
                raise ValueError(
                    f'{path}: keys are not in byte-lexicographic order '
                    f'({bad:,} inversions). Binary search would miss every '
                    f'lookup. Rebuild the index with the current merge.')
        # Newly mined recipes, appended live as workers finish. The sorted
        # array is immutable (searchsorted needs it that way), so growth goes
        # into a dict consulted alongside it. len() covers both, so MinedCache
        # grows with every newly reduced integral.
        self.extra = {}
        self.n_lookup = 0        # integrals ASKED about -- the denominator
        self.n_hit = 0
        self.n_verified = 0
        self.n_rejected = 0

    def __len__(self):
        return len(self.keys) + len(self.extra)

    def add(self, X, op, seed):
        """Append one mined recipe. Returns True if it is new."""
        if X in self.extra:
            return False
        if self._in_array(X):
            return False
        self.extra[X] = (op, seed)
        return True

    def _in_array(self, integral):
        try:
            k = np.frombuffer(pack(integral), dtype=np.uint8).view(
                np.dtype((np.void, N_IDX)))[0]
        except (ValueError, TypeError):
            return False
        i = np.searchsorted(self._flat, k)
        return i < len(self._flat) and self._flat[i] == k

    def get(self, integral):
        """Return (op, seed) for `integral`, or None. Checks live entries too."""
        got = self.extra.get(integral)
        if got is not None:
            return got
        try:
            k = np.frombuffer(pack(integral), dtype=np.uint8).view(
                np.dtype((np.void, N_IDX)))[0]
        except (ValueError, TypeError):
            return None          # component outside the packable range
        i = np.searchsorted(self._flat, k)
        if i >= len(self._flat) or self._flat[i] != k:
            return None
        v = self.vals[i]
        return int(v[0]), unpack(bytes(v[1:]))

    def _reject(self, integral, why):
        """A rejection is a DEFECT, not a statistic.

        Every recipe was verified legal when the index was built -- 0 illegal
        out of 40,402,554 records. Regenerating the same (op, seed) in the same
        environment must reproduce the same legal reduction. So a rejection can
        only mean the index and this process disagree: different prime,
        different master set, a different raw-equation strip threshold, or an
        index built against another topology.

        That is worth shouting about, because the quiet failure is worse than
        the loud one -- every lookup rejects, the hit rate reads 0%, and the
        cache looks merely useless instead of misconfigured. The first one
        prints in full; after that it is counted, and the count being nonzero
        at all is the alarm.
        """
        self.n_rejected += 1
        if self.n_rejected == 1:
            print(f"\n*** MINED CACHE MISMATCH: {why}\n"
                  f"    integral I{list(integral)}\n"
                  f"    This recipe verified when the index was built, so the "
                  f"index does not match this process.\n"
                  f"    Check SAILIR_PRIME, the master set "
                  f"(paper-masters-only), the raw-equation strip threshold, "
                  f"and the topology.\n"
                  f"    Targets fall back to workers, so results stay correct "
                  f"-- but the cache is not doing its job.\n", flush=True)
        return None

    def filter_present(self, integrals):
        """Which of `integrals` are in the index -- ONE vectorised search.

        The fold asks this of every term in the expression, so it must not be a
        per-key Python call: at 1.78M terms that is ~5 us each. Even packing in
        a Python loop costs 5.09 s measured; letting numpy convert the list of
        tuples to a 2-D array in C and doing a single searchsorted is 2.23 s for
        a bit-identical answer.
        """
        keys = integrals if isinstance(integrals, list) else list(integrals)
        if not keys:
            return set()
        a = np.array(keys, dtype=np.int16)        # list -> (n,15) array, in C
        a += OFF
        ok = ((a >= 0) & (a <= 255)).all(axis=1)  # unpackable components
        buf = np.ascontiguousarray(a.astype(np.uint8))
        flat = buf.view(np.dtype((np.void, N_IDX))).ravel()
        pos = np.searchsorted(self._flat, flat)
        np.clip(pos, 0, len(self._flat) - 1, out=pos)
        hit = ok & (self._flat[pos] == flat)
        found = {keys[j] for j in np.nonzero(hit)[0]}
        if self.extra:
            found.update(k for k in keys if k in self.extra)
        return found

    def resolve(self, integral, env, is_master, tkey, solve_ibp_for):
        """Regenerate and re-verify the reduction. Returns a rule dict, or None.

        None on a MISS is the normal case: the integral is not in the index, so
        it dispatches to a worker as usual. None after a HIT is a defect -- see
        _reject.
        """
        self.n_lookup += 1
        got = self.get(integral)
        if got is None:
            return None                      # miss: normal, not counted as bad
        self.n_hit += 1
        op, seed = got
        try:
            raw = env.get_raw_equation_cached(op, seed)
        except Exception as e:
            return self._reject(integral, f'get_raw_equation raised: {e}')
        if not raw:
            return self._reject(integral, 'identity regenerated EMPTY')
        if integral not in raw:
            return self._reject(
                integral, 'the integral is absent from its own identity')
        # the recipe is only valid if `integral` really is this identity's
        # maximal element -- otherwise solving for it is not a reduction
        if min(raw, key=tkey) != integral:
            return self._reject(
                integral, 'the integral is no longer the identity\'s maximal '
                          'element (total order or master set differs)')
        ki = tkey(integral)
        for k in raw:
            if k == integral:
                continue
            if not (tkey(k) > ki or is_master(k)):
                return self._reject(
                    integral, f'term I{list(k)} is neither lower nor a master '
                              f'(descent would break the fold)')
        rule = solve_ibp_for(raw, integral)
        if not rule:
            return self._reject(integral, 'solve_ibp_for returned nothing '
                                          '(leading coefficient vanishes mod p)')
        self.n_verified += 1
        return rule

    def stats(self):
        pct = (100.0 * self.n_verified / self.n_lookup) if self.n_lookup else 0.0
        return (f"mined cache: {len(self):,} entries; {self.n_lookup:,} looked up, "
                f"{self.n_hit:,} found, {self.n_verified:,} usable ({pct:.2f}%), "
                f"{self.n_rejected:,} rejected")


def load_if_configured():
    """Load the index named by SAILIR_RECIPE_INDEX, or return None.

    Absent or unreadable is not an error -- the campaign simply runs without
    mining, exactly as it does today.
    """
    p = os.environ.get('SAILIR_RECIPE_INDEX', '')
    if not p:
        return None
    if not os.path.exists(p):
        print(f"[recipe-index] {p} does not exist -- running without it",
              flush=True)
        return None
    try:
        idx = RecipeIndex(p)
    except Exception as e:
        print(f"[recipe-index] failed to load {p}: {e} -- running without it",
              flush=True)
        return None
    print(f"[recipe-index] loaded {len(idx):,} recipes from {p} "
          f"({(idx.keys.nbytes + idx.vals.nbytes) / 1e6:.0f} MB resident)",
          flush=True)
    return idx


def mine_path(path, env, is_master, tkey, n_idx=N_IDX):
    """Mine a finished walk's trajectory into recipes: [(X, op, seed), ...].

    Called in the WORKER, where every raw equation the walk used is already in
    env._raw_eq_cache -- so this is a few hundred cached lookups, ~7 ms, against
    a job that ran for minutes. Doing it in the orchestrator instead would mean
    re-deriving ~87 identities for each of ~7,400 results per iteration, ~51 s
    of serial work per iteration.

    X is the identity's maximal element (argmin tkey). Legality can only fail on
    an exact tie, since X is the minimum, so only ties are tested.
    """
    out = []
    for tgt, op, delta in path:
        seed = tuple(tgt[i] + delta[i] for i in range(n_idx))
        try:
            raw = env.get_raw_equation_cached(op, seed)
        except Exception:
            continue
        if not raw or len(raw) < 2:
            continue
        best_t = None
        best_k = None
        tied = None
        for k in raw:
            t = tkey(k)
            if best_t is None or t < best_t:
                best_t = t; best_k = k; tied = None
            elif t == best_t:
                tied = k
        if tied is not None and not is_master(tied):
            continue
        out.append((best_k, op, seed))
    return out
