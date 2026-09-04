"""
SAILIR action classifier.

A poly-encoder cross-attention model that, given the current expression state,
scores a variable-sized set of candidate IBP actions.

State components encoded:
    1. Expression terms (integral + coefficient)
    2. Target integral (also injected as a [TARGET] token into the Transformer)
    3. Substitution history (key integral + all replacement terms with coeffs)
    4. Sector mask (6-bit binary)
    5. Candidate actions (template index + seed integral delta)

Output: per-action logits + softmax probabilities over the valid action set.

This single module supersedes the chain earlier classifier_v1..v5 chain
from earlier iterations; the architecture and parameter names are identical to
IBPActionClassifierV5 so the published checkpoint loads without modification.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Per-component encoders (input embeddings)
# ---------------------------------------------------------------------------

class IntegralEncoder(nn.Module):
    """Encode an n-tuple integral index into a vector, plus weight features.

    n_indices is the topology-dependent tuple length (7 for trianglebox,
    11 for pentagon-box). The legacy default n_indices=7 preserves the
    behaviour of the published trianglebox checkpoint.
    """
    def __init__(self, embed_dim=64, max_index=20, min_index=-10, n_indices=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.min_index = min_index
        self.num_values = max_index - min_index + 1
        self.n_indices = n_indices

        self.position_embeds = nn.ModuleList([
            nn.Embedding(self.num_values, embed_dim // 2) for _ in range(n_indices)
        ])

        self.weight_enc = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
        )

        self.combine = nn.Sequential(
            nn.Linear(n_indices * (embed_dim // 2) + embed_dim // 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, integral):
        n = self.n_indices
        shifted = (integral - self.min_index).clamp(0, self.num_values - 1)
        orig_shape = shifted.shape[:-1]
        shifted_flat = shifted.reshape(-1, n)
        integral_flat = integral.reshape(-1, n).float()

        embeds = [self.position_embeds[i](shifted_flat[:, i]) for i in range(n)]

        sum_pos = integral_flat.clamp(min=0).sum(dim=-1, keepdim=True)
        sum_neg = (-integral_flat).clamp(min=0).sum(dim=-1, keepdim=True)
        weight_features = torch.cat([sum_pos / 10.0, sum_neg / 10.0], dim=-1)
        weight_emb = self.weight_enc(weight_features)

        combined = torch.cat(embeds + [weight_emb], dim=-1)
        return self.combine(combined).reshape(*orig_shape, self.embed_dim)


class CoefficientEncoder(nn.Module):
    """Encode finite-field coefficients with two parallel representations.

    Small values (|c| <= 31) get a learned lookup; larger values are encoded
    by (log magnitude, sign, mod-100, is-small) features.
    """
    def __init__(self, embed_dim=64, *, prime):
        super().__init__()
        self.prime = prime
        self.half_prime = prime // 2

        self.small_embed = nn.Embedding(64, embed_dim // 2)  # signed range -32..31
        self.large_embed = nn.Sequential(
            nn.Linear(4, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
        )
        self.combine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # FULL LOOKUP TABLE over the field, behind a zero-init gate.
        #
        # The features above are an integer intuition applied to a residue:
        # "log magnitude" and "mod 100" mean little in GF(p), where p-1 is -1
        # rather than a large number. Measured on the corrupt cull corpus,
        # 43.4% of nonzero coefficients miss the |signed| <= 31 lookup and are
        # described by those features, and rational reconstruction rescues only
        # 56.8% of those occurrences -- so they are not a complete answer.
        #
        # forward() is a function of ONE discrete value in [0, prime), so any
        # encoder of it is exactly representable by an Embedding(prime,
        # embed_dim). That table is therefore the most expressive coefficient
        # encoder that can exist here, and at prime=1009 it costs 129k params
        # (+1.5% of the model) with NO change to activation width -- which
        # matters because this runs at B*A*T ~ 129k terms per batch.
        #
        # Zero-init gate so a warm start from a checkpoint without the table is
        # bit-identical at init; the table has to earn its way in, exactly as
        # eq_gate does for the equation branch.
        self.use_table = os.environ.get('SAILIR_COEFF_TABLE') == '1'
        if self.use_table:
            # prime is a REQUIRED kwarg and must never be defaulted: a default
            # 2^31-1 against a trained 1009 silently flipped 8.2% of top-1
            # predictions for two months.
            self.coeff_table = nn.Embedding(prime, embed_dim)
            nn.init.normal_(self.coeff_table.weight, std=0.02)
            self.coeff_table_gate = nn.Parameter(torch.zeros(1))

    def forward(self, coeff):
        orig_shape = coeff.shape
        coeff_flat = coeff.reshape(-1).float()

        signed = torch.where(coeff_flat > self.half_prime,
                             coeff_flat - self.prime, coeff_flat)

        small_idx = (signed.long() + 32).clamp(0, 63)
        small_emb = self.small_embed(small_idx)

        abs_val = torch.abs(signed)
        large_features = torch.stack([
            torch.log1p(abs_val) / 20.0,
            (signed >= 0).float(),
            (abs_val % 100).float() / 100.0,
            (abs_val < 100).float(),
        ], dim=-1)
        large_emb = self.large_embed(large_features)

        combined = torch.cat([small_emb, large_emb], dim=-1)
        out = self.combine(combined)
        if self.use_table:
            # Residues are already in [0, prime); the modulo is a cheap guard
            # against an out-of-range index becoming a silent CUDA assert.
            idx = coeff.reshape(-1).long().remainder(self.prime)
            out = out + self.coeff_table_gate * self.coeff_table(idx)
        return out.reshape(*orig_shape, -1)


class ActionEncoder(nn.Module):
    """Encode an action (template index, seed integral delta) into a vector."""
    def __init__(self, embed_dim=128, n_ibp_ops=9, max_index=20, min_index=-10,
                 n_indices=7):
        super().__init__()
        self.ibp_embed = nn.Embedding(n_ibp_ops, embed_dim // 2)
        self.delta_enc = IntegralEncoder(embed_dim // 2, max_index, min_index,
                                          n_indices=n_indices)
        self.combine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, ibp_op, delta):
        ibp_emb = self.ibp_embed(ibp_op)
        delta_emb = self.delta_enc(delta)
        return self.combine(torch.cat([ibp_emb, delta_emb], dim=-1))


# ---------------------------------------------------------------------------
# State encoders
# ---------------------------------------------------------------------------

class FullSubstitutionEncoder(nn.Module):
    """Encode the complete substitution history (key integral + replacement terms).

    Each substitution maps an integral (the "key") to a linear combination of
    other integrals.  We encode both sides, pool the replacement set with
    learned-query attention, combine, then run the per-substitution embeddings
    through a positional-encoded Transformer + a final attention pool.
    """
    def __init__(self, embed_dim=256, max_index=20, min_index=-10, *, prime,
                 n_heads=4, n_layers=2, max_subs=50, max_replacement_terms=20,
                 n_indices=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_replacement_terms = max_replacement_terms
        self.n_indices = n_indices

        self.key_integral_enc = IntegralEncoder(embed_dim, max_index, min_index,
                                                 n_indices=n_indices)
        self.replacement_integral_enc = IntegralEncoder(embed_dim // 2, max_index, min_index,
                                                         n_indices=n_indices)
        self.replacement_coeff_enc = CoefficientEncoder(embed_dim // 2, prime=prime)

        self.replacement_term_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
        )

        self.replacement_pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.replacement_pool_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

        self.sub_combine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
        )

        self.pos_encoding = nn.Parameter(torch.randn(max_subs, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.final_pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.final_pool_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def encode_single_substitution(self, key_integral, replacement_integrals, replacement_coeffs, replacement_mask):
        batch_size = key_integral.size(0)

        key_emb = self.key_integral_enc(key_integral)

        repl_int_emb = self.replacement_integral_enc(replacement_integrals)
        repl_coeff_emb = self.replacement_coeff_enc(replacement_coeffs)
        repl_term_emb = self.replacement_term_proj(torch.cat([repl_int_emb, repl_coeff_emb], dim=-1))

        query = self.replacement_pool_query.expand(batch_size, -1, -1)
        key_pad_mask = ~replacement_mask if replacement_mask is not None else None

        if replacement_mask is not None:
            has_terms = replacement_mask.any(dim=1)
            if not has_terms.all():
                repl_pooled = torch.zeros(batch_size, self.embed_dim, device=key_integral.device)
                if has_terms.any():
                    valid_idx = has_terms.nonzero(as_tuple=True)[0]
                    valid_pooled, _ = self.replacement_pool_attn(
                        query[valid_idx], repl_term_emb[valid_idx], repl_term_emb[valid_idx],
                        key_padding_mask=key_pad_mask[valid_idx],
                    )
                    repl_pooled[valid_idx] = valid_pooled.squeeze(1)
            else:
                repl_pooled, _ = self.replacement_pool_attn(
                    query, repl_term_emb, repl_term_emb, key_padding_mask=key_pad_mask,
                )
                repl_pooled = repl_pooled.squeeze(1)
        else:
            repl_pooled, _ = self.replacement_pool_attn(query, repl_term_emb, repl_term_emb)
            repl_pooled = repl_pooled.squeeze(1)

        return self.sub_combine(torch.cat([key_emb, repl_pooled], dim=-1))

    def forward(self, sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask):
        batch_size, max_subs, max_repl, _ = sub_repl_ints.shape
        device = sub_mask.device

        if not sub_mask.any():
            return torch.zeros(batch_size, self.embed_dim, device=device)

        n = self.n_indices
        flat_sub_embs = self.encode_single_substitution(
            sub_keys.view(batch_size * max_subs, n),
            sub_repl_ints.view(batch_size * max_subs, max_repl, n),
            sub_repl_coeffs.view(batch_size * max_subs, max_repl),
            sub_repl_mask.view(batch_size * max_subs, max_repl),
        )
        all_sub_embs = flat_sub_embs.view(batch_size, max_subs, self.embed_dim)
        all_sub_embs = all_sub_embs * sub_mask.unsqueeze(-1).float()
        all_sub_embs = all_sub_embs + self.pos_encoding[:max_subs].unsqueeze(0)

        attn_mask = ~sub_mask
        encoded = self.transformer(all_sub_embs, src_key_padding_mask=attn_mask)

        result = torch.zeros(batch_size, self.embed_dim, device=device)
        has_subs = sub_mask.any(dim=1)
        if has_subs.any():
            valid_idx = has_subs.nonzero(as_tuple=True)[0]
            query = self.final_pool_query.expand(valid_idx.numel(), -1, -1)
            valid_pooled, _ = self.final_pool_attn(
                query, encoded[valid_idx], encoded[valid_idx],
                key_padding_mask=attn_mask[valid_idx],
            )
            result[valid_idx] = valid_pooled.squeeze(1)
        return result


class SectorEncoder(nn.Module):
    """Encode an n-bit sector mask via per-bit embeddings + MLP.

    n_denominators is the topology-dependent denominator count
    (6 for trianglebox, 8 for pentagon-box).
    """
    def __init__(self, embed_dim=256, n_denominators=6):
        super().__init__()
        self.n_denominators = n_denominators
        per_bit = embed_dim // n_denominators + 1
        self.position_embeddings = nn.ModuleList([
            nn.Embedding(2, per_bit) for _ in range(n_denominators)
        ])
        self.proj = nn.Sequential(
            nn.Linear(n_denominators * per_bit, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, sector_mask):
        n = self.n_denominators
        pos_embs = [self.position_embeddings[i](sector_mask[:, i].long()) for i in range(n)]
        return self.proj(torch.cat(pos_embs, dim=-1))


class TransformerExpressionEncoderWithTarget(nn.Module):
    """Transformer over the expression sequence [CLS] [TARGET] term_1 ... term_N.

    No positional encoding -- the order of expression terms is physically
    meaningless.  The CLS and TARGET tokens are distinguished from regular
    terms by their embedding content (TARGET uses an integral-only embedding,
    regular terms use integral+coefficient).
    """
    def __init__(self, embed_dim=256, max_index=20, min_index=-10, *, prime,
                 n_heads=4, n_layers=2, max_terms=512, n_indices=7):
        super().__init__()
        self.embed_dim = embed_dim

        self.integral_enc = IntegralEncoder(embed_dim // 2, max_index, min_index,
                                             n_indices=n_indices)
        self.coeff_enc = CoefficientEncoder(embed_dim // 2, prime=prime)
        self.target_integral_enc = IntegralEncoder(embed_dim, max_index, min_index,
                                                    n_indices=n_indices)

        self.term_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
        )
        self.target_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim),
        )
        self.target_output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, integrals, coeffs, target_integral, mask=None, return_per_term=False):
        batch_size, max_terms, _ = integrals.shape

        int_emb = self.integral_enc(integrals)
        coeff_emb = self.coeff_enc(coeffs)
        term_emb = self.term_proj(torch.cat([int_emb, coeff_emb], dim=-1))

        target_emb = self.target_proj(self.target_integral_enc(target_integral)).unsqueeze(1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, target_emb, term_emb], dim=1)

        if mask is not None:
            special_mask = torch.ones(batch_size, 2, dtype=torch.bool, device=mask.device)
            attn_mask = ~torch.cat([special_mask, mask], dim=1)
        else:
            attn_mask = None

        encoded = self.transformer(seq, src_key_padding_mask=attn_mask)

        cls_pooled = self.cls_output_proj(encoded[:, 0])
        target_pooled = self.target_output_proj(encoded[:, 1])
        per_term = encoded[:, 2:]

        if return_per_term:
            return cls_pooled, target_pooled, per_term
        return cls_pooled, target_pooled


# ---------------------------------------------------------------------------
# Cross-attention scorer (variable-size action set)
# ---------------------------------------------------------------------------

class CrossAttentionScorer(nn.Module):
    """Score each candidate action by attending it to per-term expression embeddings.

    Multi-layer, multi-head cross-attention with residual + FFN blocks between
    layers, followed by an MLP that combines the attended action representation
    with the global state vector to produce per-action logits.  Invalid
    actions are masked to -inf in the logits.
    """
    def __init__(self, embed_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.state_proj = nn.Linear(embed_dim, embed_dim)
        self.action_proj = nn.Linear(embed_dim, embed_dim)
        self.expr_term_proj = nn.Linear(embed_dim, embed_dim)

        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.cross_attn_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim), nn.Dropout(0.1),
            ) for _ in range(n_layers)
        ])
        self.cross_attn_ffn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])

        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, state_emb, action_emb, expr_terms, expr_mask, action_mask):
        batch_size, n_actions, _ = action_emb.shape

        state_proj = self.state_proj(state_emb)
        action_proj = self.action_proj(action_emb)
        expr_proj = self.expr_term_proj(expr_terms)
        key_pad_mask = ~expr_mask if expr_mask is not None else None

        attended = action_proj
        for i in range(len(self.cross_attn_layers)):
            attn_out, _ = self.cross_attn_layers[i](
                attended, expr_proj, expr_proj, key_padding_mask=key_pad_mask,
            )
            attended = self.cross_attn_norms[i](attended + attn_out)
            attended = self.cross_attn_ffn_norms[i](attended + self.cross_attn_ffn[i](attended))

        state_expanded = state_proj.unsqueeze(1).expand(-1, n_actions, -1)
        logits = self.scorer(torch.cat([state_expanded, attended], dim=-1)).squeeze(-1)
        return logits.masked_fill(~action_mask, float('-inf'))


# ---------------------------------------------------------------------------
# Full classifier
# ---------------------------------------------------------------------------

class IBPActionClassifier(nn.Module):
    """SAILIR action classifier.

    Combines the four state encoders + the cross-attention action scorer into
    a single end-to-end model.  Forward returns (logits, softmax_probs) over
    the valid action set.

    Parameter layout (submodule names) matches the IBPActionClassifierV5 class
    used to train the published checkpoint; ``load_state_dict`` on the
    published weights works without modification.
    """
    def __init__(self, embed_dim=256, n_heads=4, n_expr_layers=2, n_cross_layers=2,
                 n_subs_layers=2, *, prime, n_indices=7, n_denominators=6,
                 n_ibp_ops=9, **kwargs):
        super().__init__()
        self.prime = prime
        self.embed_dim = embed_dim
        self.n_indices = n_indices
        self.n_denominators = n_denominators

        # kwargs are passed only to sub-encoders that accept them; we split
        # n_ibp_ops here because only ActionEncoder uses it.
        self.expr_enc = TransformerExpressionEncoderWithTarget(
            embed_dim, prime=prime, n_heads=n_heads, n_layers=n_expr_layers,
            n_indices=n_indices, **kwargs,
        )
        self.subs_enc = FullSubstitutionEncoder(
            embed_dim, n_heads=n_heads, n_layers=n_subs_layers, prime=prime,
            n_indices=n_indices, **kwargs,
        )
        self.sector_enc = SectorEncoder(embed_dim, n_denominators=n_denominators)
        self.action_enc = ActionEncoder(embed_dim, n_indices=n_indices,
                                         n_ibp_ops=n_ibp_ops, **kwargs)

        # State combine: cls + target + subs + sector -> embed_dim
        self.state_combine = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.scorer = CrossAttentionScorer(embed_dim, n_heads, n_cross_layers)

    def forward(self, expr_integrals, expr_coeffs, expr_mask,
                sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask,
                action_ibp_ops, action_deltas, action_mask,
                sector_mask, target_integral):
        cls_pooled, target_pooled, expr_terms = self.expr_enc(
            expr_integrals, expr_coeffs, target_integral, expr_mask, return_per_term=True,
        )
        subs_emb = self.subs_enc(sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask)
        sector_emb = self.sector_enc(sector_mask)

        state_emb = self.state_combine(
            torch.cat([cls_pooled, target_pooled, subs_emb, sector_emb], dim=-1)
        )
        action_emb = self.action_enc(action_ibp_ops, action_deltas)
        logits = self.scorer(state_emb, action_emb, expr_terms, expr_mask, action_mask)
        return logits, F.softmax(logits, dim=-1)

    def predict(self, *args, **kwargs):
        logits, _ = self.forward(*args, **kwargs)
        return logits.argmax(dim=-1)


# Backwards-compatibility alias: the published checkpoint was saved under this
# class name.  Loading with ``torch.load(...).get('model_class', ...)`` still
# works.
IBPActionClassifierV5 = IBPActionClassifier
