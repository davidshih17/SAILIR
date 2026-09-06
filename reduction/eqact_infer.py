"""Inference-side equation resolution for the `eqact` model.

WHY THIS EXISTS. At training time each row's resolved equations came from a
PRECOMPUTE (eqpre*/). At search time that is useless: the beam visits states no
corpus contains, so every candidate action's equation must be resolved from the
CURRENT substitution store, every step, every beam state.

PARITY IS THE WHOLE POINT. This calls the SAME `EqResolver` the training
dataloader used, so the equations the model sees at search time are produced by
the identical code path -- not a reimplementation that happens to look right.
The one thing that must be mirrored by hand is `_truncate_expr`: training cut
the expression to the top SAILIR_EXPR_TERMS by total lex weight, and the beam
builds full expressions, so without the same cut the model meets inputs it never
saw.

COST, measured at depth 2400 (store = 2400 substitutions, beam width 40):
    store -> dict conversion   5.6 ms/store   (2.3 us/substitution)
    raw equations (cached fn)  62 us/action   <- dominates
    apply store                 3 us/action
    added per step             ~2.6-2.8 s  against ~9.5 s for a whole step
i.e. roughly +30%. NOTE the operator sample behind the per-action figures is
biased small (median 5 / max 7 raw terms), so treat +30% as a FLOOR.

The store conversion is cached on the store OBJECT's identity, which is sound
for the same reason _V9_SMW_CACHE is: beam states share their store object until
a substitution is added, and apply_action_v5 builds a NEW dict for the child.
"""
import os

import numpy as np
import torch

_B = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'

# Lazily built: importing the training dataset module at import time would drag
# torch DataLoader machinery into every worker that never uses eqact.
_STATE = {}


def is_eqact(model):
    """True for an IBPActionClassifierEqAct. Duck-typed on the gate rather than
    isinstance, so this module never has to import the model class."""
    return model is not None and getattr(model, 'eq_gate', None) is not None


def _lazy(env, ibp_t, li_t, n_idx):
    if _STATE.get('res') is None:
        import sys
        for p in (_B, f'{_B}/training', f'{_B}/reduction'):
            if p not in sys.path:
                sys.path.insert(0, p)
        from eqact_dataset import EqResolver, MAX_EQ_TERMS, MAX_EXPR_TERMS
        _STATE['res'] = EqResolver(ibp_t, li_t, n_idx, env=env)
        _STATE['T'] = MAX_EQ_TERMS
        _STATE['E'] = MAX_EXPR_TERMS
        print(f'[eqact-infer] EqResolver ready: T={MAX_EQ_TERMS} '
              f'EXPR_TERMS={MAX_EXPR_TERMS}', flush=True)
    return _STATE['res']


# Store-object identity -> unpacked {key: {term: coeff}}. Bounded: only the
# current beam's stores are reachable, and an entry dies with its state.
_STORE_CACHE = {}
_STORE_ORDER = []
_STORE_CAP = int(os.environ.get('SAILIR_EQ_STORE_CACHE', '256'))


def _unpack(resolved_subs, registry):
    """PackedEq store -> the dict form EqResolver expects, cached on identity."""
    if not resolved_subs:
        return {}
    k = id(resolved_subs)
    hit = _STORE_CACHE.get(k)
    if hit is not None:
        return hit[1]
    out = {kk: (vv.to_dict(registry) if hasattr(vv, 'to_dict') else vv)
           for kk, vv in resolved_subs.items()}
    # Keep a reference to the store alongside the dict: without it the id()
    # could be recycled by a NEW store at the same address and silently return
    # another state's substitutions.
    _STORE_CACHE[k] = (resolved_subs, out)
    _STORE_ORDER.append(k)
    while len(_STORE_ORDER) > _STORE_CAP:
        _STORE_CACHE.pop(_STORE_ORDER.pop(0), None)
    return out


def truncate_expr(batch, weight_fn):
    """Cut expr tensors to the top MAX_EXPR_TERMS by total lex weight, IN PLACE.

    Mirrors train_eqact._truncate_expr. Training truncated with top_by_weight
    (numpy lexsort on (w1, w2)); the same helper is imported here rather than
    rewritten, because a different tie-break would silently feed the model a
    different expression than it trained on.
    """
    E = _STATE.get('E', 0)
    if not E or E <= 0:
        return batch
    import sys
    if f'{_B}/training' not in sys.path:
        sys.path.insert(0, f'{_B}/training')
    from eqact_dataset import top_by_weight
    ei, ec, em = batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask']
    if ei.shape[1] <= E:
        return batch
    B_, _, N = ei.shape
    oi = torch.zeros(B_, E, N, dtype=ei.dtype)
    oc = torch.zeros(B_, E, dtype=ec.dtype)
    om = torch.zeros(B_, E, dtype=em.dtype)
    ein, ecn = ei.cpu().numpy(), ec.cpu().numpy()
    for b in range(B_):
        n = int(em[b].sum())
        if n == 0:
            continue
        a, c = ein[b, :n], ecn[b, :n]
        if n > E:
            a, c = top_by_weight(a, c, E)
        k = len(a)
        oi[b, :k] = torch.from_numpy(np.ascontiguousarray(a))
        oc[b, :k] = torch.from_numpy(np.ascontiguousarray(c))
        om[b, :k] = True
    batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'] = oi, oc, om
    return batch


def build_eq_tensors(chunk, registry, env, ibp_t, li_t, n_idx, A_pad, device):
    """Resolved-equation tensors for one model chunk.

    chunk: list of (expr, resolved_subs, valid_actions, target_sector, target)
    Returns dict with eq_integrals (B,A,T,N) int8, eq_coeffs (B,A,T) int16,
    eq_mask (B,A,T) bool, eq_pivot (B,A,N) int8 -- the dtypes the collate ships
    and encode_equations casts on device.
    """
    res = _lazy(env, ibp_t, li_t, n_idx)
    T = _STATE['T']
    B_ = len(chunk)
    ei = torch.zeros(B_, A_pad, T, n_idx, dtype=torch.int8)
    ec = torch.zeros(B_, A_pad, T, dtype=torch.int16)
    em = torch.zeros(B_, A_pad, T, dtype=torch.bool)
    pv = torch.zeros(B_, A_pad, n_idx, dtype=torch.int8)
    for b, item in enumerate(chunk):
        expr, rs, valid, _sec, target = item[0], item[1], item[2], item[3], item[4]
        if not valid:
            continue
        n = min(len(valid), A_pad)
        ops = [v[0] for v in valid[:n]]
        deltas = [v[1] for v in valid[:n]]
        store = _unpack(rs, registry)
        r_i, r_c, r_m, r_p = res.row(target, ops, deltas, store)
        a = min(n, r_i.shape[0])
        ei[b, :a] = r_i[:a] if torch.is_tensor(r_i) else torch.as_tensor(r_i[:a])
        ec[b, :a] = r_c[:a] if torch.is_tensor(r_c) else torch.as_tensor(r_c[:a])
        em[b, :a] = r_m[:a] if torch.is_tensor(r_m) else torch.as_tensor(r_m[:a])
        pv[b, :a] = r_p[:a] if torch.is_tensor(r_p) else torch.as_tensor(r_p[:a])
    return {'eq_integrals': ei.to(device), 'eq_coeffs': ec.to(device),
            'eq_mask': em.to(device), 'eq_pivot': pv.to(device)}
