"""No-subs variant of the SAILIR action classifier.

Mirrors `IBPActionClassifier` from `sailir/classifier.py` but drops the
`FullSubstitutionEncoder` entirely. The state vector that gets broadcast to
the cross-attention scorer is built from `cls_pooled`, `target_pooled`, and
`sector_emb` only (3 channels instead of 4), so `state_combine` is
`3*embed_dim -> embed_dim`.

Motivation: the trained pentagonbox-10x classifier was found to be
content-insensitive to subs (zero-content random subs and shuffled subs both
produce ~bit-identical predictions). The subs encoder is ~38% of total
parameters and most of the cost beyond `expr_enc`. This variant strips it out
so we can quantify size, memory, and step-time savings, and provide a
candidate replacement architecture to retrain.

Shares the encoder building blocks (TransformerExpressionEncoderWithTarget,
SectorEncoder, ActionEncoder, CrossAttentionScorer) with the full model by
importing them from `sailir.classifier`. The shared modules are not duplicated
or modified.

forward() takes a strict subset of the full model's arguments — the sub_*
tensors are NOT in the signature. Drive selection through the
`--model_variant` flag in train_classifier.py rather than by trying to load
weights across variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# train_classifier.py loads this module after inserting `sailir/` onto
# sys.path, in which case `classifier` is the right name. The eval/benchmark
# scripts add the project root, in which case `sailir.classifier` is the
# discoverable name. Handle both.
try:
    from sailir.classifier import (
        TransformerExpressionEncoderWithTarget,
        SectorEncoder,
        ActionEncoder,
        CrossAttentionScorer,
        IntegralEncoder,
    )
except ImportError:
    from classifier import (
        TransformerExpressionEncoderWithTarget,
        SectorEncoder,
        ActionEncoder,
        CrossAttentionScorer,
        IntegralEncoder,
    )


class IBPActionClassifierNoSubs(nn.Module):
    """SAILIR action classifier without the subs encoder.

    Constructor accepts the same kwargs as `IBPActionClassifier` for symmetry;
    `n_subs_layers` is accepted but ignored.
    """

    def __init__(self, embed_dim=256, n_heads=4, n_expr_layers=2, n_cross_layers=2,
                 n_subs_layers=2, *, prime, n_indices=7, n_denominators=6,
                 n_ibp_ops=9, use_start_target=False, use_value_head=False,
                 score_activation='softmax', **kwargs):
        super().__init__()
        self.prime = prime
        # How forward()'s SECOND return value is produced -- the beam consumes
        # it as `action_prob` and accumulates log(prob) into the path score.
        #   'softmax'  : the historical behaviour. DEFAULT, so every existing
        #                checkpoint and the whole current beam path are
        #                bit-identical.
        #   'sigmoid'  : for models trained with SAILIR_LOSS=bce. Required for
        #                those, because softmax divides by sum_j exp(z_j) over
        #                the action set and so reintroduces at INFERENCE exactly
        #                the action-count dependence the BCE objective removes
        #                during training. NOTE the returned values then no
        #                longer sum to 1, so state.score stops being a
        #                log-likelihood and becomes a sum of independent log
        #                confidences -- comparable within a model, NOT across a
        #                bce model and a ce model.
        if score_activation not in ('softmax', 'sigmoid'):
            raise ValueError(f'score_activation must be softmax|sigmoid, '
                             f'got {score_activation!r}')
        self.score_activation = score_activation
        self.embed_dim = embed_dim
        self.n_indices = n_indices
        self.n_denominators = n_denominators
        del n_subs_layers  # explicitly noted as ignored

        self.expr_enc = TransformerExpressionEncoderWithTarget(
            embed_dim, prime=prime, n_heads=n_heads, n_layers=n_expr_layers,
            n_indices=n_indices, **kwargs,
        )
        self.sector_enc = SectorEncoder(embed_dim, n_denominators=n_denominators)
        self.action_enc = ActionEncoder(embed_dim, n_indices=n_indices,
                                         n_ibp_ops=n_ibp_ops, **kwargs)

        # State combine: cls + target + sector (+ start target) -> embed_dim.
        # EXP 1 (use_start_target): the model is given the CURRENT target (the
        # top of the bucket, recomputed every step) but never the START target
        # T, even though the task is defined relative to T -- "active work"
        # means at-or-above T in the total order, so T is the floor of the
        # active region. Without it, two states with identical expressions but
        # different T look the same while having different work remaining.
        # Widening this layer breaks checkpoint shape compatibility, so
        # load_v7_into() below copies the old weights and ZERO-INITS the new
        # slice: the augmented model starts numerically identical to v7 and
        # learns to use T from there.
        self.use_start_target = use_start_target
        n_chan = 4 if use_start_target else 3
        if use_start_target:
            self.start_target_enc = IntegralEncoder(
                embed_dim, n_indices=n_indices)
            self.start_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                nn.LayerNorm(embed_dim),
            )
        self.state_combine = nn.Sequential(
            nn.Linear(embed_dim * n_chan, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.scorer = CrossAttentionScorer(embed_dim, n_heads, n_cross_layers)

        # EXP 2 (use_value_head): predict how many steps remain until the
        # bucket drains. The beam has no cost-to-go signal at all -- measured
        # on the truth paths, the active count rises as often as it falls and
        # max_w12 is unchanged on 71-86% of steps -- so it ranks survivors by
        # quantities a correct reduction does not monotonically improve.
        # state_emb is already computed for the action scorer, so this is one
        # small head on an existing vector, not an extra encoder pass.
        self.use_value_head = use_value_head
        if use_value_head:
            self.value_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
            )

    def forward(self, expr_integrals, expr_coeffs, expr_mask,
                sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask,
                action_ibp_ops, action_deltas, action_mask,
                sector_mask, target_integral, start_target_integral=None):
        # sub_* positional args accepted for call-site parity with
        # IBPActionClassifier; they are NOT read.
        del sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask

        cls_pooled, target_pooled, expr_terms = self.expr_enc(
            expr_integrals, expr_coeffs, target_integral, expr_mask, return_per_term=True,
        )
        sector_emb = self.sector_enc(sector_mask)

        chans = [cls_pooled, target_pooled, sector_emb]
        if self.use_start_target:
            if start_target_integral is None:
                raise ValueError(
                    "use_start_target=True but start_target_integral was not "
                    "passed — the caller must thread T through. Silently "
                    "substituting the current target would train the model on "
                    "a different input than it sees at search time.")
            # Embed T with its OWN encoder rather than re-running expr_enc
            # against T. Re-encoding would cost a second full pass of the
            # expression transformer (3.0M params, the largest module) on every
            # beam state at every step; this is an embedding lookup plus a small
            # projection. T is a single integral with no expression context to
            # attend over, so the transformer pass would buy nothing.
            chans.append(self.start_proj(
                self.start_target_enc(start_target_integral)))

        state_emb = self.state_combine(torch.cat(chans, dim=-1))
        action_emb = self.action_enc(action_ibp_ops, action_deltas)
        logits = self.scorer(state_emb, action_emb, expr_terms, expr_mask, action_mask)
        if self.use_value_head:
            # 3-tuple ONLY when the head is enabled, so every existing caller
            # doing `logits, _ = model(...)` keeps working untouched.
            return logits, self._scores(logits), self.value_head(state_emb).squeeze(-1)
        return logits, self._scores(logits)

    def _scores(self, logits):
        """Second return value: per-action scores for the beam."""
        if self.score_activation == 'sigmoid':
            # Padded slots carry -inf, and sigmoid(-inf) = 0, matching what
            # softmax gives them. No masking needed.
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)

    def predict(self, *args, **kwargs):
        logits, _ = self.forward(*args, **kwargs)
        return logits.argmax(dim=-1)
