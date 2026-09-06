"""Verify the SAILIR_LOSS=bce objective does what it claims.

Run with SAILIR_LOSS=bce so _action_loss takes the bce branch.

Checks:
  1. LOSS INVARIANCE to action count -- a state with 1000 actions / 18 correct
     and one with 100 actions / 2 correct produce the SAME loss when their
     per-action logit quality is the same. This is the property the doubly
     normalised form is for.
  2. A naive flat-mean BCE does NOT have that property (shows the
     normalisation is load-bearing, not decorative).
  3. No NaN from the -inf padded slots.
  4. Gradients are finite.
  5. sigmoid scores are action-count independent; softmax scores are not.
"""
import os, sys
from pathlib import Path
assert os.environ.get('SAILIR_LOSS') == 'bce', "run with SAILIR_LOSS=bce"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from train_classifier import _action_loss

NEG_INF = float('-inf')


def make_state(n_act, n_pos, max_act, pos_logit=1.0, neg_logit=-3.0):
    """One state: n_act real actions of which n_pos are correct, padded to max_act.

    Logits are ASYMMETRIC on purpose. With +/-z, BCE(z,1)==BCE(-z,0)==softplus(-z),
    so positive and negative errors coincide and even a flat mean looks
    invariant -- the test would prove nothing.
    """
    logits = torch.full((max_act,), NEG_INF)
    logits[:n_pos] = pos_logit
    logits[n_pos:n_act] = neg_logit
    action_mask = torch.zeros(max_act, dtype=torch.bool); action_mask[:n_act] = True
    label_mask = torch.zeros(max_act, dtype=torch.bool);  label_mask[:n_pos] = True
    return logits, action_mask, label_mask


def batch_of(states, max_act):
    L, A, Y = zip(*states)
    return (torch.stack(L).requires_grad_(True),
            {'action_mask': torch.stack(A), 'label_mask': torch.stack(Y),
             'labels': torch.zeros(len(states), dtype=torch.long)})


ok = True
MAX = 1000

# ---- 1. invariance ----------------------------------------------------------
lg_a, b_a = batch_of([make_state(1000, 18, MAX)], MAX)
lg_b, b_b = batch_of([make_state(100,   2, MAX)], MAX)
La = _action_loss(lg_a, b_a).item()
Lb = _action_loss(lg_b, b_b).item()
print(f"1. loss(1000 act/18 pos) = {La:.8f}")
print(f"   loss( 100 act/ 2 pos) = {Lb:.8f}")
print(f"   |diff|                = {abs(La-Lb):.3e}")
if abs(La - Lb) > 1e-6:
    print("   FAIL: not invariant to action count"); ok = False
else:
    print("   PASS: invariant")

# extra pairs, incl. the pathological ends of the measured range
for (na, npos), (nb, nposb) in [((1000, 1), (74, 1)), ((500, 86), (120, 20))]:
    x, bx = batch_of([make_state(na, npos, MAX)], MAX)
    y, by = batch_of([make_state(nb, nposb, MAX)], MAX)
    d = abs(_action_loss(x, bx).item() - _action_loss(y, by).item())
    tag = "PASS" if d < 1e-6 else "FAIL"
    print(f"   {tag}: ({na}/{npos}) vs ({nb}/{nposb})  |diff|={d:.3e}")
    ok &= d < 1e-6

# ---- 2. naive flat mean is NOT invariant ------------------------------------
def flat_bce(logits, batch):
    act = batch['action_mask']
    safe = torch.where(act, logits, torch.zeros_like(logits))
    tgt = batch['label_mask'].to(safe.dtype)
    el = F.binary_cross_entropy_with_logits(safe, tgt, reduction='none')
    return (el * act).sum() / act.sum()

fa, fb = flat_bce(lg_a, b_a).item(), flat_bce(lg_b, b_b).item()
print(f"\n2. flat-mean BCE: 1000/18 = {fa:.6f}   100/2 = {fb:.6f}   "
      f"|diff|={abs(fa-fb):.3e}")
print("   PASS: flat mean IS count-sensitive (so the normalisation matters)"
      if abs(fa - fb) > 1e-4 else "   (unexpected: flat mean looks invariant)")

# ---- 3/4. NaN + gradients ---------------------------------------------------
lg, b = batch_of([make_state(1000, 18, MAX), make_state(100, 2, MAX),
                  make_state(74, 1, MAX)], MAX)
loss = _action_loss(lg, b)
loss.backward()
print(f"\n3. loss finite: {torch.isfinite(loss).item()}  value={loss.item():.6f}")
print(f"4. grads finite: {torch.isfinite(lg.grad).all().item()}   "
      f"(padded-slot grads all zero: {(lg.grad[~b['action_mask']] == 0).all().item()})")
ok &= torch.isfinite(loss).item() and torch.isfinite(lg.grad).all().item()

# ---- 5. score activation ----------------------------------------------------
print("\n5. score for an action with logit=2.0, as the action set grows:")
for n in (10, 100, 1000):
    z = torch.full((n,), -2.0); z[0] = 2.0
    print(f"   n_act={n:5d}   softmax={F.softmax(z, -1)[0]:.6f}   "
          f"sigmoid={torch.sigmoid(z)[0]:.6f}")
print("   -> softmax collapses with action count; sigmoid is constant.")

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
