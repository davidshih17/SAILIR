#!/usr/bin/env python
"""Symmetry engine for the 3-loop GRAVITY family (GR) — the gravity analog of
symmetry_engine.py (pentagonbox), with two structural differences:

1. LINEAR (eikonal) propagators. D5/D6/D7 and four of the five ISPs are LINEAR
   in the loop momenta (e.g. D6 = -2 k1.u2), not squared momenta. Under a
   relabeling a linear denominator can map to MINUS another denominator — a
   per-slot coefficient the pentagonbox image function never sees (all its
   propagators are quadratic). Consumers MUST apply den-row coefficients
   (see canonicalize_GR.image_signed_dens).

2. NO per-record momentum reconstruction. 395/551 Kira records are placeholders
   (no stored momentum map), and Kira's `ing` "fixed" entries can be fake
   (the pentagonbox lesson). Instead we enumerate the WHOLE candidate symmetry
   group offline: integer affine maps  k_i -> sum_j a_ij k_j + b_i q  with
   a in {-1,0,1}^{3x3}, det(a) = +-1, b in {-1,0,1}^3 (the observed vocabulary;
   the few coefficient-2 maps live in records WITH real substitution strings and
   are derived directly). Every such map is a valid change of variables
   (|det|=1), so every derived transform is an exact value identity — Kira's
   records only tell us which maps are USEFUL per sector. A map "realizes" a
   record when, on the record's source-sector denominators, its rows are
   single-term den->den with zero constant and land exactly at ing[g].

The 15 scalar-product unknowns (3 loops; externals q, u1, u2 with q.q = -1,
u1.u1 = u2.u2 = 1, u1.u2 = y, q.u1 = q.u2 = 0) are in bijection with the 15
propagators, so A (15x15) is invertible and  M = A S A^-1 ,
c = A s0 + a0 - M a0  for the sp-space action (S, s0) of a map.

Clean-row detection is done at TWO independent evaluation points (p,y) =
(1009,31) and (2003,17) to kill accidental mod-p zeros; stored transforms are
at the production point (1009, 31).

Run this file to build + gate; it saves  results/gr_transforms.pkl :
  {'by_sector': {src_sector: [(M, c), ...]},   # M[i]={j:co}, c[i]=const, mod 1009
   'gates': {...}}
"""
import os, sys, itertools, pickle
import numpy as np

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)

N_IND, N_DEN, N_LOOP = 15, 10, 3
# PROD/CHECK are overridable so the store can be rebuilt at the MODEL's prime.
# The p101 model was trained on a corpus generated entirely at 101; the
# orchestrator refuses --use-symmetry unless --prime matches the store's point,
# because a transform's coefficients are rationals reduced mod p and must
# combine with IBP coefficients at the SAME prime. Defaults unchanged, so an
# unset environment reproduces the validated (1009, 31) store exactly.
#
# CHECK is the independent second point that kills accidental mod-p zeros. It
# MUST stay a different prime from PROD -- at p=101 spurious cancellation is far
# likelier than at 1009, so this check matters MORE for a small prime, not less.
PROD = (int(os.environ.get('SAILIR_SYM_PRIME', 1009)),
        int(os.environ.get('SAILIR_SYM_Y', 31)))
CHECK = (int(os.environ.get('SAILIR_SYM_CHECK_PRIME', 2003)),
         int(os.environ.get('SAILIR_SYM_CHECK_Y', 17)))
if PROD[0] == CHECK[0]:
    raise SystemExit(f'PROD and CHECK must use DIFFERENT primes '
                     f'(got {PROD[0]} for both) -- the second point exists to '
                     f'detect accidental mod-p zeros and is useless at the same p.')

# ---- scalar-product basis: index 0..14 ----
# kk: (0,0)(0,1)(0,2)(1,1)(1,2)(2,2) -> 0..5 ; kq: 6..8 ; ku1: 9..11 ; ku2: 12..14
KK = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}
KQ = lambda i: 6 + i
KU1 = lambda i: 9 + i
KU2 = lambda i: 12 + i


def kk_idx(i, j):
    return KK[(i, j) if i <= j else (j, i)]


def build_A(p, y):
    """A[i] = sp-coefficients of propagator i; a0[i] = constant. All mod p.
    Propagators (integralfamilies.yaml order, 0-based):
      quadratic (mom m -> m.m): built from momentum coefficient vectors
      linear: explicit sp combinations."""
    A = np.zeros((N_IND, N_IND), dtype=np.int64)
    a0 = np.zeros(N_IND, dtype=np.int64)

    def quad(row, kvec, qco):
        """(sum_i kvec[i]*k_i + qco*q)^2 -> sp coeffs + const (q.q=-1)."""
        for i in range(3):
            for j in range(i, 3):
                co = kvec[i] * kvec[j] if i == j else 2 * kvec[i] * kvec[j]
                A[row, kk_idx(i, j)] = (A[row, kk_idx(i, j)] + co) % p
            A[row, KQ(i)] = (A[row, KQ(i)] + 2 * kvec[i] * qco) % p
        a0[row] = (a0[row] - qco * qco) % p           # q.q = -1

    quad(0, (1, 0, 0), 0)          # D1  k1^2
    quad(1, (0, 1, 0), 0)          # D2  k2^2
    quad(2, (0, 0, 1), 0)          # D3  k3^2
    quad(3, (1, 1, 1), -1)         # D4  (k1+k2+k3-q)^2
    A[4, KU1(0)] = 2; A[4, KU1(1)] = 2            # D5  2(k1+k2).u1
    A[5, KU2(0)] = (-2) % p                        # D6  -2 k1.u2
    A[6, KU2(0)] = A[6, KU2(1)] = A[6, KU2(2)] = (-2) % p   # D7 -2(k1+k2+k3).u2
    quad(7, (1, 1, 0), 0)          # D8  (k1+k2)^2
    quad(8, (1, 1, 0), -1)         # D9  (k1+k2-q)^2
    quad(9, (0, 1, 1), 0)          # D10 (k2+k3)^2
    A[10, KU2(1)] = 1              # D11 k2.u2
    A[11, KU1(0)] = 1              # D12 k1.u1
    A[12, KU1(2)] = 1              # D13 k3.u1
    A[13, kk_idx(0, 2)] = 1        # D14 k1.k3
    A[14, KQ(0)] = 1               # D15 k1.q
    return A % p, a0 % p


def inv_mod(Amat, p):
    """Matrix inverse mod prime p via Gauss-Jordan (small, exact)."""
    n = Amat.shape[0]
    M = np.concatenate([Amat % p, np.eye(n, dtype=np.int64)], axis=1)
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r, col] % p), None)
        assert piv is not None, f"A singular mod {p}"
        M[[col, piv]] = M[[piv, col]]
        M[col] = (M[col] * pow(int(M[col, col]), p - 2, p)) % p
        for r in range(n):
            if r != col and M[r, col]:
                M[r] = (M[r] - M[r, col] * M[col]) % p
    return M[:, n:]


def sp_action(a, b, p, y, swap_u=False, neg_q=False):
    """(S, s0): action of the substitution  k_i -> sum_j a[i][j] k_j + b[i] q,
    optionally composed with the EXTERNAL involutions  u1 <-> u2  (swap_u) and
    q -> -q (neg_q); both preserve every kinematic invariant (q.q=-1, u.u=1,
    u1.u2=y, q.u=0) so they are exact value identities when the propagators map
    into the family. S[new_sp_row, old_sp_col]; value(new sp) = S @ v + s0.

    The identity being built (verify_map checks it numerically):
      D_g(phi(k); q, u1, u2)  ==  sum_h M[g][h] D_h(k; q', u1', u2') + c_g
    with phi(k)_i = sum_j a_ij k_j + b_i q (REAL q) and (q', u') the relabeled
    externals on the TARGET side. So S = S_phi @ R^{-1}: S_phi is the pure
    loop-substitution action; R^{-1} relabels the COLUMN basis (kq columns get
    s_q, ku1/ku2 columns swap) and leaves all constants untouched — the q -> -q
    flip multiplies the kq columns of EVERY row (including the kk rows' shift
    cross-terms) and never the q.q constants. Getting this wrong produced false
    I = -I zeros (caught by the numeric gate, 2026-07-17)."""
    S = np.zeros((N_IND, N_IND), dtype=np.int64)
    s0 = np.zeros(N_IND, dtype=np.int64)
    sq = -1 if neg_q else 1
    U1c, U2c = (KU2, KU1) if swap_u else (KU1, KU2)     # column target for u-rows
    for i in range(3):
        for j in range(i, 3):
            r = kk_idx(i, j)
            for l in range(3):
                for m in range(3):
                    S[r, kk_idx(l, m)] += a[i][l] * a[j][m]
                S[r, KQ(l)] += sq * (a[i][l] * b[j] + b[i] * a[j][l])
            s0[r] -= b[i] * b[j]                       # q.q = -1 (R-invariant)
        r = KQ(i)
        for l in range(3):
            S[r, KQ(l)] += sq * a[i][l]
        s0[r] -= b[i]                                   # q.q = -1 (R-invariant)
        for l in range(3):
            S[KU1(i), U1c(l)] += a[i][l]                # ku columns swap under R
            S[KU2(i), U2c(l)] += a[i][l]
    return S % p, s0 % p


def transform_of(a, b, ctx, swap_u=False, neg_q=False):
    """(M, c) for the map at evaluation context ctx = (p, A, Ainv, a0, y)."""
    p, A, Ainv, a0, y = ctx
    S, s0 = sp_action(a, b, p, y, swap_u, neg_q)
    M = (A @ S @ Ainv) % p
    c = (A @ s0 + a0 - M @ a0) % p
    return M, c


def den_signature(M, c, p):
    """Per den slot g: target slot if row g is single-term den->den with zero
    const (coefficient may be any nonzero value — linear dens flip signs), else
    -1. Returns int8 array length N_DEN."""
    sig = np.full(N_DEN, -1, dtype=np.int8)
    for g in range(N_DEN):
        nz = np.nonzero(M[g] % p)[0]
        if len(nz) == 1 and nz[0] < N_DEN and c[g] % p == 0:
            sig[g] = nz[0]
    return sig


def make_ctx(p, y):
    A, a0 = build_A(p, y)
    return (p, A, inv_mod(A, p), a0, y)


def parse_subst(loop_substs):
    """Kira loop_subst strings -> (a 3x3, b 3) integer map, or None if any
    placeholder. Vocabulary: integer combinations of k1,k2,k3,q."""
    import sympy as sp
    k = sp.symbols('k1 k2 k3'); q = sp.Symbol('q')
    loc = {'k1': k[0], 'k2': k[1], 'k3': k[2], 'q': q}
    a = [[0] * 3 for _ in range(3)]; b = [0] * 3
    for lhs, rhs in loop_substs:
        if rhs == "placeholder":
            return None
        i = int(lhs[1]) - 1
        e = sp.expand(sp.sympify(rhs, locals=loc))
        for j in range(3):
            a[i][j] = int(e.coeff(k[j]))
        b[i] = int(e.coeff(q))
        assert sp.expand(e - sum(a[i][j] * k[j] for j in range(3)) - b[i] * q) == 0
    det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
           - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
           + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    assert det in (1, -1), f"non-unimodular subst det={det}: {loop_substs}"
    return tuple(tuple(r) for r in a), tuple(b)


def to_dicts(M, c, p):
    Md = {i: {int(j): int(M[i, j] % p) for j in np.nonzero(M[i] % p)[0]}
          for i in range(N_IND)}
    Md = {i: r for i, r in Md.items() if r}
    cd = {i: int(c[i] % p) for i in range(N_IND) if c[i] % p}
    return Md, cd


# momentum coefficient vectors (k1,k2,k3,q) of the QUADRATIC propagators
QUAD_MOM = {0: (1, 0, 0, 0), 1: (0, 1, 0, 0), 2: (0, 0, 1, 0), 3: (1, 1, 1, -1),
            7: (1, 1, 0, 0), 8: (1, 1, 0, -1), 9: (0, 1, 1, 0)}
# LINEAR propagators: (u-vector tag, loop-coefficient vector, overall factor)
# D5 = 2(k1+k2).u1, D6 = -2 k1.u2, D7 = -2(k1+k2+k3).u2
LIN_DEN = {4: ('u1', (1, 1, 0), 2), 5: ('u2', (1, 0, 0), -2), 6: ('u2', (1, 1, 1), -2)}


def targeted_search(present, ing, swap_u=False, neg_q=False):
    """Solve for integer maps (a, b) realizing `ing` on `present` dens under the
    external variant (swap_u, neg_q): for each sign assignment over the den
    constraints, the equations
      quadratic g:  T(m_g) = eps_g * m_{ing[g]}      (4 eqs: k1,k2,k3,q)
      linear g:     f_g * (sum_i v_i T(k_i)).u' = eps_g * (target linear form)
    are linear in the 12 unknowns (a 3x3, b 3). Solve exactly (Fractions), keep
    integer solutions with det(a) = +-1. Returns list of (a, b)."""
    from fractions import Fraction
    from itertools import product as iprod
    sq = -1 if neg_q else 1
    swap_tag = {'u1': 'u2', 'u2': 'u1'} if swap_u else {'u1': 'u1', 'u2': 'u2'}
    cons = []                                    # (kind, g, h) in fixed order
    for g in present:
        h = ing[g]
        if g in QUAD_MOM:
            if h not in QUAD_MOM:
                return []
            cons.append(('q', g, h))
        else:
            if h not in LIN_DEN or LIN_DEN[h][0] != swap_tag[LIN_DEN[g][0]]:
                return []                        # u-tag must match under the variant
            cons.append(('l', g, h))
    sols = []
    for signs in iprod((1, -1), repeat=len(cons)):
        rowsM, rhs = [], []                      # over unknowns x = (a11..a33, b1..b3)
        for (kind, g, h), eps in zip(cons, signs):
            if kind == 'q':
                mg, mh = QUAD_MOM[g], QUAD_MOM[h]
                for j in range(3):               # k_j component
                    row = [0] * 12
                    for i in range(3):
                        row[3 * i + j] = mg[i]
                    rowsM.append(row); rhs.append(eps * mh[j])
                row = [0] * 12                   # q component: phi(m_g) has
                # q-coeff  sum_i (m_g)_i b_i + (m_g)_q ; the target momentum at
                # relabeled externals has q-coeff  eps * s_q * (m_h)_q
                for i in range(3):
                    row[9 + i] = mg[i]
                rowsM.append(row); rhs.append(eps * sq * mh[3] - mg[3])
            else:
                _, vg, fg = LIN_DEN[g]
                _, vh, fh = LIN_DEN[h]
                for j in range(3):               # k_j.u component
                    row = [0] * 12
                    for i in range(3):
                        row[3 * i + j] = fg * vg[i]
                    rowsM.append(row); rhs.append(eps * fh * vh[j])
        # exact Gaussian elimination
        A = [[Fraction(x) for x in row] + [Fraction(r)]
             for row, r in zip(rowsM, rhs)]
        n, m = len(A), 12
        piv_cols, r = [], 0
        for cidx in range(m):
            piv = next((k for k in range(r, n) if A[k][cidx] != 0), None)
            if piv is None:
                continue
            A[r], A[piv] = A[piv], A[r]
            A[r] = [x / A[r][cidx] for x in A[r]]
            for k in range(n):
                if k != r and A[k][cidx] != 0:
                    fac = A[k][cidx]
                    A[k] = [x - fac * y for x, y in zip(A[k], A[r])]
            piv_cols.append(cidx); r += 1
        if any(all(A[k][cidx] == 0 for cidx in range(m)) and A[k][m] != 0
               for k in range(r, n)):
            continue                             # inconsistent
        free = [cidx for cidx in range(m) if cidx not in piv_cols]
        if len(free) > 4:
            continue                             # too underdetermined to enumerate
        for fv in iprod((-2, -1, 0, 1, 2), repeat=len(free)):
            x = [Fraction(0)] * m
            for cidx, v in zip(free, fv):
                x[cidx] = Fraction(v)
            for k, cidx in enumerate(piv_cols):
                x[cidx] = A[k][m] - sum(A[k][j] * x[j] for j in free)
            if any(v.denominator != 1 for v in x):
                continue
            xi = [int(v) for v in x]
            a = (tuple(xi[0:3]), tuple(xi[3:6]), tuple(xi[6:9]))
            b = tuple(xi[9:12])
            det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                   - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                   + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
            if det in (1, -1):
                sols.append((a, b))
    return sols


# ---------------------------------------------------------------------------
# NUMERIC GROUND-TRUTH GATE: explicit GF(p) vector kinematics. Metric
# diag(+,-,-,-,-,-); externals constructed exactly (u1.u1=u2.u2=1, u1.u2=y,
# q.q=-1, q.u=0); loops random. Every stored transform must satisfy
#   D_g(phi(k); q,u)  ==  sum_h M[g][h] D_h(k; q',u') + c_g   (all 15 slots)
# on several random draws — this sees SIGN errors den signatures cannot.
# ---------------------------------------------------------------------------
_METRIC = np.array([1, -1, -1, -1, -1, -1], dtype=np.int64)


def _vdot(u, v, p):
    return int((u * v * _METRIC).sum() % p)


def _kin_vectors(p, y, rng):
    s2 = (y * y - 1) % p
    s = next((r for r in range(p) if r * r % p == s2), None)
    assert s is not None, f"y^2-1 not a QR mod {p} — choose another split"
    u1 = np.array([1, 0, 0, 0, 0, 0], dtype=np.int64)
    u2 = np.array([y, s, 0, 0, 0, 0], dtype=np.int64)
    q = np.array([0, 0, 1, 0, 0, 0], dtype=np.int64)
    assert (_vdot(u1, u1, p), _vdot(u2, u2, p), _vdot(u1, u2, p),
            _vdot(q, q, p), _vdot(q, u1, p), _vdot(q, u2, p)) == \
        (1, 1, y % p, (-1) % p, 0, 0)
    return q, u1, u2


def _props_at(k1, k2, k3, qv, u1v, u2v, p):
    def sq(v):
        return _vdot(v, v, p)
    return [
        sq(k1), sq(k2), sq(k3), sq((k1 + k2 + k3 - qv) % p),
        2 * _vdot((k1 + k2) % p, u1v, p) % p,
        -2 * _vdot(k1, u2v, p) % p,
        -2 * _vdot((k1 + k2 + k3) % p, u2v, p) % p,
        sq((k1 + k2) % p), sq((k1 + k2 - qv) % p), sq((k2 + k3) % p),
        _vdot(k2, u2v, p), _vdot(k1, u1v, p), _vdot(k3, u1v, p),
        _vdot(k1, k3, p), _vdot(k1, qv, p),
    ]


def verify_map(a, b, su, nq, ctx, rng, n_trials=4):
    """True iff the derived (M, c) is an exact identity on random GF(p) vectors."""
    p = ctx[0]; y = ctx[4]
    M, c = transform_of(a, b, ctx, su, nq)
    q, u1, u2 = _kin_vectors(p, y, rng)
    qp = (-q) % p if nq else q
    u1p, u2p = (u2, u1) if su else (u1, u2)
    for _ in range(n_trials):
        ks = [np.array([rng.randrange(p) for _ in range(6)], dtype=np.int64)
              for _ in range(3)]
        D = _props_at(*ks, qp, u1p, u2p, p)          # target side: relabeled ext
        kp = [(sum(a[i][j] * ks[j] for j in range(3)) + b[i] * q) % p
              for i in range(3)]
        Dp = _props_at(*kp, q, u1, u2, p)            # source side: original ext
        for g in range(N_IND):
            rhs = (sum(int(M[g, j]) * D[j] for j in range(N_IND)) + int(c[g])) % p
            if Dp[g] % p != rhs:
                return False
    return True


def build_and_gate():
    from sailir.symmetries import parse_symmetries
    SYMDIR = os.path.join(ROOT, "topology_input/gravity3L/kira_validate/sectormappings/GR")
    recs = (parse_symmetries(os.path.join(SYMDIR, "sectorSymmetries"), N_IND, n_loops=3)
            + parse_symmetries(os.path.join(SYMDIR, "sectorRelations"), N_IND, n_loops=3))
    print(f"records: {len(recs)}")

    ctxP, ctxC = make_ctx(*PROD), make_ctx(*CHECK)

    # ---- enumerate the {-1,0,1} grid, det +-1 ----
    mats = [m for m in itertools.product((-1, 0, 1), repeat=9)
            if (lambda a: a[0] * (a[4] * a[8] - a[5] * a[7])
                - a[1] * (a[3] * a[8] - a[5] * a[6])
                + a[2] * (a[3] * a[7] - a[4] * a[6]))(m) in (1, -1)]
    shifts = list(itertools.product((-1, 0, 1), repeat=3))
    print(f"det+-1 a-matrices: {len(mats)}  x  q-shifts: {len(shifts)} "
          f"= {len(mats)*len(shifts)} candidate maps")

    # den signatures at BOTH points; a slot is clean only if clean at both and
    # both agree on the target (kills accidental mod-p zeros). Four external
    # variants: (swap_u, neg_q) — u1<->u2 and q->-q are exact invariant-
    # preserving involutions the loop maps cannot produce.
    VARIANTS = [(False, False), (True, False), (False, True), (True, True)]
    n_maps = len(mats) * len(shifts) * len(VARIANTS)
    sigs = np.full((n_maps, N_DEN), -1, dtype=np.int8)
    maps = []
    t = 0
    for m in mats:
        a = (m[0:3], m[3:6], m[6:9])
        for b in shifts:
            for su, nq in VARIANTS:
                MP, cP = transform_of(a, b, ctxP, su, nq)
                sP = den_signature(MP, cP, PROD[0])
                if (sP >= 0).any():
                    MC, cC = transform_of(a, b, ctxC, su, nq)
                    sC = den_signature(MC, cC, CHECK[0])
                    agree = (sP == sC)
                    sP[~agree] = -1
                sigs[t] = sP
                maps.append((a, b, su, nq))
                t += 1
    useful = int(((sigs >= 0).sum(axis=1) > 0).sum())
    print(f"maps with >=1 clean den slot (both points): {useful}/{n_maps}")

    # ---- match records; EVERY stored transform passes the numeric gate ----
    import random as _random
    _rng = _random.Random(20260717)
    by_sector = {}
    stats = dict(real_total=0, real_gate_fail=0, ph_total=0, ph_matched=0,
                 ph_unmatched=[], numeric_fail=0)
    match_counts = []
    for r in recs:
        present = [g for g in range(N_DEN) if r.source_sector >> g & 1]
        ing = r.ing
        ab = parse_subst(list(r.loop_substs))
        if ab is not None:
            # real substitution: derive directly; gate den behavior vs ing
            stats['real_total'] += 1
            MP, cP = transform_of(*ab, ctxP)
            sP = den_signature(MP, cP, PROD[0])
            ok = all(sP[g] == ing[g] for g in present)
            if ok and not verify_map(ab[0], ab[1], False, False, ctxP, _rng):
                stats['numeric_fail'] += 1
                ok = False
                print(f"  NUMERIC FAIL (real) sector {r.source_sector}")
            if not ok:
                stats['real_gate_fail'] += 1
                print(f"  REAL-GATE FAIL sector {r.source_sector} "
                      f"ing={[ing[g] for g in present]} sig={[int(sP[g]) for g in present]}")
                continue
            Md, cd = to_dicts(MP, cP, PROD[0])
            by_sector.setdefault(r.source_sector, []).append((Md, cd))
        else:
            # placeholder: find ALL grid maps realizing ing on the present dens
            stats['ph_total'] += 1
            cond = np.ones(n_maps, dtype=bool)
            for g in present:
                cond &= (sigs[:, g] == ing[g])
            idx = np.nonzero(cond)[0]
            cands = [maps[t] for t in idx]
            if not cands:
                # coefficient-2 maps: targeted constraint solve per external
                # variant, then VERIFY the den signature at both points
                found = []
                for su, nq in VARIANTS:
                    for a, b in targeted_search(present, ing, su, nq):
                        MP, cP = transform_of(a, b, ctxP, su, nq)
                        sP = den_signature(MP, cP, PROD[0])
                        MC, cC = transform_of(a, b, ctxC, su, nq)
                        sC = den_signature(MC, cC, CHECK[0])
                        if all(sP[g] == ing[g] and sC[g] == ing[g] for g in present):
                            found.append((a, b, su, nq))
                cands = found
                if cands:
                    stats['ph_targeted'] = stats.get('ph_targeted', 0) + 1
            # numeric ground-truth gate — drops sign-consistent-but-false maps
            verified = [m4 for m4 in cands if verify_map(*m4, ctxP, _rng)]
            stats['numeric_fail'] += len(cands) - len(verified)
            match_counts.append(len(verified))
            if not verified:
                stats['ph_unmatched'].append((r.source_sector,
                                              tuple(int(ing[g]) for g in present)))
                continue
            stats['ph_matched'] += 1
            for a, b, su, nq in verified:
                MP, cP = transform_of(a, b, ctxP, su, nq)
                Md, cd = to_dicts(MP, cP, PROD[0])
                by_sector.setdefault(r.source_sector, []).append((Md, cd))

    # dedup per sector
    total = 0
    for s in list(by_sector):
        seen, out = set(), []
        for (Md, cd) in by_sector[s]:
            key = (tuple(sorted((i, tuple(sorted(rr.items()))) for i, rr in Md.items())),
                   tuple(sorted(cd.items())))
            if key not in seen:
                seen.add(key); out.append((Md, cd))
        by_sector[s] = out
        total += len(out)

    if match_counts:
        mc = sorted(match_counts)
        print(f"placeholder match multiplicity: p10={mc[len(mc)//10]} "
              f"p50={mc[len(mc)//2]} p90={mc[9*len(mc)//10]} max={mc[-1]}")
    print(f"real records: {stats['real_total']} (gate fails {stats['real_gate_fail']})")
    print(f"placeholder records: {stats['ph_total']} matched {stats['ph_matched']} "
          f"(of which via targeted solve: {stats.get('ph_targeted', 0)}), "
          f"unmatched {len(stats['ph_unmatched'])}")
    print(f"numeric-gate rejections (sign-consistent but FALSE maps): "
          f"{stats['numeric_fail']}")
    for s, ig in stats['ph_unmatched'][:20]:
        print(f"    UNMATCHED sector {s} ing_on_dens={ig}")
    print(f"transform store: {sum(1 for _ in by_sector)} sectors, {total} distinct transforms")

    out = {"by_sector": by_sector, "gates": stats,
           "prod_point": PROD, "check_point": CHECK}
    _out = os.environ.get('SAILIR_SYM_OUT', 'results/gr_transforms.pkl')
    with open(os.path.join(ROOT, _out), "wb") as f:
        pickle.dump(out, f)
    print(f"saved -> {_out}  (prod_point={PROD} check_point={CHECK})")
    ok = stats['real_gate_fail'] == 0 and not stats['ph_unmatched']
    print("ALL PASS" if ok else "GATE INCOMPLETE (see above)")
    return out


if __name__ == "__main__":
    build_and_gate()
