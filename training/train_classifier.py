"""Training script for the SAILIR IBP action classifier.

Trains on packed-tensor shards produced by `../data-gen/preprocess_to_tensors.py`.

Two model variants are supported:
  * `full`   — `IBPActionClassifier` with the subs encoder (see
               `sailir/classifier.py`). The committed reference checkpoint
               `checkpoints/pentagonbox_10x_loop_100/best_model.pt` was trained
               with this variant.
  * `nosubs` — `IBPActionClassifierNoSubs` with the subs path removed (see
               `sailir/classifier_nosubs.py`). Recommended going forward:
               matches `full` accuracy with ~40% fewer params and ~12% faster
               steps (see `README.md`).

DDP launching: this script is `torchrun`-aware. Set `LOCAL_RANK`/`RANK`/`WORLD_SIZE`
(set automatically by `torchrun` / `srun --gpus-per-task=1`) and run as usual.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'sailir'))
from classifier import IBPActionClassifier
from classifier_nosubs import IBPActionClassifierNoSubs

MODEL_CLASSES = {
    'full': IBPActionClassifier,
    'nosubs': IBPActionClassifierNoSubs,
}


class PackedDatasetV5(Dataset):
    """Dataset using packed format with offsets, including target and subs_raw."""

    def __init__(self, data):
        self.data = data
        self.n_samples = len(data['labels'])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        d = self.data

        expr_start, expr_end = d['expr_offsets'][idx].item(), d['expr_offsets'][idx + 1].item()
        action_start, action_end = d['action_offsets'][idx].item(), d['action_offsets'][idx + 1].item()
        # Corpora packed with include_subs=False carry no substitution fields at
        # all (they are dead weight for the nosubs model, 44.6% of the bytes).
        _has_subs = 'sub_offsets' in d
        if _has_subs:
            sub_start, sub_end = d['sub_offsets'][idx].item(), d['sub_offsets'][idx + 1].item()

        return {
            'expr_integrals': d['expr_integrals'][expr_start:expr_end],
            'expr_coeffs': d['expr_coeffs'][expr_start:expr_end],
            'sub_integrals': (d['sub_integrals'][sub_start:sub_end] if _has_subs
                              else torch.zeros(0, d['target_integrals'].shape[1],
                                               dtype=torch.int8)),
            'subs_raw': d['subs_raw'][idx] if _has_subs else [],
            'action_ibp_ops': d['action_ibp_ops'][action_start:action_end],
            'action_deltas': d['action_deltas'][action_start:action_end],
            'sector_mask': d['sector_masks'][idx],
            'target_integral': d['target_integrals'][idx],  # Target integral
            # EXP 1: START target T. Falls back to the per-step target so
            # packed files written before EXP 1 still load.
            'start_target_integral': (d['start_target_integrals'][idx]
                                     if 'start_target_integrals' in d
                                     else d['target_integrals'][idx]),
            # EXP 2: steps-to-go; 0 = unknown -> masked out of the value loss
            'steps_remaining': (d['steps_remaining'][idx]
                                if 'steps_remaining' in d else 0),
            'label': d['labels'][idx],
            # MULTI-LABEL: every correct action for this state. Sliced from the
            # flat store; empty when the shard predates the field, and the
            # collate falls back to the single label so old shards still train.
            'valid_label_idxs': (
                d['valid_label_idxs'][
                    int(d['valid_label_offsets'][idx]):
                    int(d['valid_label_offsets'][idx + 1])]
                if 'valid_label_offsets' in d else None),
        }


class TokenBudgetBatchSampler(torch.utils.data.Sampler):
    """Size-bucketed batch sampler: groups sample indices (sorted by action
    count) so that len(batch) * max_num_valid_actions_in_batch <= token_budget
    AND len(batch) <= max_batch. Exact softmax over every candidate action —
    no subsampling — with padded-batch memory bounded by construction (a 33k-
    action sample simply rides in a batch of 1). Batch ORDER is reshuffled per
    epoch (set_epoch); batch composition is fixed (standard bucketing)."""

    def __init__(self, num_actions, token_budget, max_batch, seed=0):
        import random as _random
        self._random = _random
        self.token_budget = token_budget
        self.seed = seed
        self.epoch = 0
        order = sorted(range(len(num_actions)), key=lambda i: int(num_actions[i]))
        self.batches = []
        cur, cur_max = [], 0
        for i in order:
            n = int(num_actions[i])
            m = max(cur_max, n)
            if cur and ((len(cur) + 1) * m > token_budget or len(cur) >= max_batch):
                self.batches.append(cur)
                cur, cur_max = [i], n
            else:
                cur.append(i)
                cur_max = m
        if cur:
            self.batches.append(cur)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        b = list(self.batches)
        self._random.Random(self.seed + self.epoch).shuffle(b)
        return iter(b)

    def __len__(self):
        return len(self.batches)


MAX_REPLACEMENT_TERMS = 20  # Max replacement terms per substitution


def make_collate_fn(n_indices, n_denominators):
    """Build a topology-aware collate_fn closure."""
    def collate_fn(samples):
        return _collate(samples, n_indices, n_denominators)
    return collate_fn


# SAILIR_LOSS=set switches from hard cross-entropy to the partial-label
# objective. Read once at import, not per batch.
_SET_LOSS = os.environ.get('SAILIR_LOSS', 'ce') == 'set'
_SOFT_LOSS = os.environ.get('SAILIR_LOSS', 'ce') == 'soft'
_BCE_LOSS = os.environ.get('SAILIR_LOSS', 'ce') == 'bce'
if os.environ.get('SAILIR_LOSS', 'ce') not in ('ce', 'set', 'soft', 'bce'):
    raise SystemExit(f"unknown SAILIR_LOSS={os.environ['SAILIR_LOSS']!r} "
                     f"(ce | set | soft | bce)")


def _action_loss(logits, batch):
    """Cross-entropy, or the multi-label ('set') objective.

    There are ~18 equally correct actions per state -- measured, and first /
    last / random picks all reach a successful reduction -- so hard
    cross-entropy against ONE of them actively penalises the model for
    choosing any of the other ~17. Three training runs have now shown that the
    choice of tie-break alone is worth ~20 points of val top1 (old label 87.4,
    minsumw 67.1, lex worse), on a metric that does not measure what the search
    needs.

    The set loss is  -log( sum_{i in C} softmax(logits)_i )  -- maximise the
    total probability mass on ANY correct action. Properties that matter here:
      * no arbitrary tie-break survives in the target;
      * it optimises exactly the quantity the beam cares about, "is the top
        action a reducing one", rather than conformity to one convention;
      * the model may concentrate on whichever candidate it finds easiest,
        which is the brittleness that appears to sink a hard lex label -- there
        the model must find THE lexicographic minimum, and a near miss is not a
        similar action.
    Computed as logsumexp over the candidate logits minus logsumexp over all,
    which is numerically stable and never materialises the softmax.
    """
    if _SOFT_LOSS:
        # SOFT-LABEL CROSS-ENTROPY: uniform target over the ~18 equally-correct
        # actions. Ported from RL_amplitudes_claude/src/train_sft_5pt.py:128-131,
        # where the identical problem (a set of equivalent actions, no canonical
        # one) is already solved this way.
        #
        # WHY THIS AND NOT 'set'. The set loss -log(sum_{i in C} p_i) is
        # satisfied by ANY SINGLE label reaching 1.0 -- it constrains only the
        # TOTAL mass on C, never its distribution across C. Measured on
        # rands_ep25: top_prob median 1.0000, saturated >0.9999 on 41 of 55
        # steps, with label_prob_sum=1.000000. The model became confidently
        # committed to one arbitrary correct row, which is optimal under that
        # objective and useless to a beam: at the step where it finally erred,
        # 37 closure rows shared 3e-06 of the mass, so none were rankable.
        #
        # Soft-label CE is minimised only when p MATCHES the uniform target, so
        # every correct action carries ~1/|C|. Those rows then populate the
        # top-K the beam expands instead of sitting six orders of magnitude
        # below a confidently-wrong pick.
        # NaN TRAP, hit on the first run: the scorer returns
        # `logits.masked_fill(~action_mask, -inf)` (classifier.py:415), so
        # PADDED action slots carry -inf and log_softmax gives -inf there. The
        # soft target is 0 at those slots, and 0 * -inf = NaN -- loss=nan from
        # batch 1 in both arms. The 'set' loss never hit this because logsumexp
        # handles -inf cleanly (exp(-inf)=0).
        # torch.where keeps the padded terms out of the product entirely rather
        # than multiplying and repairing afterwards, so no NaN is ever created
        # (nan_to_num after the fact would still poison the backward pass).
        mask = batch['label_mask']
        soft = mask.to(logits.dtype)
        soft = soft / soft.sum(dim=-1, keepdim=True).clamp(min=1.0)
        logp = F.log_softmax(logits, dim=-1)
        contrib = torch.where(soft > 0, soft * logp, torch.zeros_like(logp))
        return -contrib.sum(dim=-1).mean()
    if _BCE_LOSS:
        # INDEPENDENT PER-ACTION BCE: every valid action -> 1, every other -> 0.
        # Unlike softmax the outputs are uncoupled, so the model can be
        # confident on ALL ~n_pos correct actions at once instead of having to
        # split one unit of probability mass between them.
        #
        # DOUBLY NORMALISED, so the objective does not depend on how many
        # actions a state happens to offer:
        #   mean within positives  -> independent of n_pos
        #   mean within negatives  -> independent of n_neg
        #   equal 1/2 on each side -> independent of their RATIO. This is a
        #     PER-STATE pos_weight = n_neg_i/n_pos_i. A single global
        #     pos_weight is wrong almost everywhere: measured on this corpus
        #     the per-state ratio spans ~11:1 to ~999:1 around a 109:1 mean
        #     (training/measure_label_imbalance.py).
        #   mean over states       -> independent of state size (74..1000 here)
        #
        # NaN TRAP -- the same one the soft loss records hitting on its first
        # run. The scorer returns logits.masked_fill(~action_mask, -inf)
        # (classifier.py:415) and BCEWithLogits computes -x*target, so on a
        # padded slot -(-inf)*0 = nan. Substitute a finite value BEFORE the
        # BCE; nan_to_num afterwards would still poison the backward pass.
        act = batch['action_mask']
        pos = batch['label_mask'] & act
        neg = (~batch['label_mask']) & act
        safe = torch.where(act, logits, torch.zeros_like(logits))
        l_pos = F.binary_cross_entropy_with_logits(
            safe, torch.ones_like(safe), reduction='none')
        l_neg = F.binary_cross_entropy_with_logits(
            safe, torch.zeros_like(safe), reduction='none')
        n_pos = pos.sum(dim=1)
        n_neg = neg.sum(dim=1)
        mean_pos = (l_pos * pos).sum(dim=1) / n_pos.clamp(min=1)
        mean_neg = (l_neg * neg).sum(dim=1) / n_neg.clamp(min=1)
        # A state with no positives (or no negatives) contributes only the side
        # it actually has, rather than averaging in a spurious 0.
        w_pos = (n_pos > 0).to(logits.dtype)
        w_neg = (n_neg > 0).to(logits.dtype)
        per_state = (w_pos * mean_pos + w_neg * mean_neg) / (w_pos + w_neg).clamp(min=1)
        return per_state.mean()
    if not _SET_LOSS:
        return F.cross_entropy(logits, batch['labels'])
    mask = batch['label_mask']
    neg_inf = torch.finfo(logits.dtype).min
    picked = logits.masked_fill(~mask, neg_inf)
    return -(torch.logsumexp(picked, dim=-1)
             - torch.logsumexp(logits, dim=-1)).mean()


def _collate(samples, n_indices, n_denominators):
    """Pad variable-length tensors to batch max and stack."""
    batch_size = len(samples)

    max_expr = max(len(s['expr_integrals']) for s in samples)
    max_sub = max(len(s['subs_raw']) for s in samples) if any(len(s['subs_raw']) > 0 for s in samples) else 1
    max_action = max(len(s['action_ibp_ops']) for s in samples)

    expr_integrals = torch.zeros(batch_size, max_expr, n_indices, dtype=torch.long)
    expr_coeffs = torch.zeros(batch_size, max_expr, dtype=torch.long)
    expr_mask = torch.zeros(batch_size, max_expr, dtype=torch.bool)

    # Batched substitution tensors
    sub_keys = torch.zeros(batch_size, max_sub, n_indices, dtype=torch.long)
    sub_repl_ints = torch.zeros(batch_size, max_sub, MAX_REPLACEMENT_TERMS, n_indices, dtype=torch.long)
    sub_repl_coeffs = torch.zeros(batch_size, max_sub, MAX_REPLACEMENT_TERMS, dtype=torch.long)
    sub_repl_mask = torch.zeros(batch_size, max_sub, MAX_REPLACEMENT_TERMS, dtype=torch.bool)
    sub_mask = torch.zeros(batch_size, max_sub, dtype=torch.bool)

    action_ibp_ops = torch.zeros(batch_size, max_action, dtype=torch.long)
    action_deltas = torch.zeros(batch_size, max_action, n_indices, dtype=torch.long)
    action_mask = torch.zeros(batch_size, max_action, dtype=torch.bool)

    sector_masks = torch.zeros(batch_size, n_denominators, dtype=torch.long)
    target_integrals = torch.zeros(batch_size, n_indices, dtype=torch.long)
    start_target_integrals = torch.zeros(batch_size, n_indices, dtype=torch.long)
    steps_remaining = torch.zeros(batch_size, dtype=torch.float)
    labels = torch.zeros(batch_size, dtype=torch.long)
    # MULTI-LABEL target aligned with the logits: True at every action that is
    # a correct choice for this state. Built here (not in the loss) so it is
    # padded to max_action exactly like action_mask.
    label_mask = torch.zeros(batch_size, max_action, dtype=torch.bool)

    for i, s in enumerate(samples):
        n_expr = len(s['expr_integrals'])
        n_action = len(s['action_ibp_ops'])

        expr_integrals[i, :n_expr] = s['expr_integrals'].long()
        expr_coeffs[i, :n_expr] = s['expr_coeffs'].long()
        expr_mask[i, :n_expr] = True

        # Process subs_raw into batched tensors
        subs = s['subs_raw']
        for j, sub in enumerate(subs[:max_sub]):
            sub_mask[i, j] = True
            sub_keys[i, j] = torch.tensor(sub[0], dtype=torch.long)
            replacement = sub[1]
            n_repl = min(len(replacement), MAX_REPLACEMENT_TERMS)
            for r in range(n_repl):
                sub_repl_ints[i, j, r] = torch.tensor(replacement[r][0], dtype=torch.long)
                sub_repl_coeffs[i, j, r] = replacement[r][1]
                sub_repl_mask[i, j, r] = True

        action_ibp_ops[i, :n_action] = s['action_ibp_ops'].long()
        action_deltas[i, :n_action] = s['action_deltas'].long()
        action_mask[i, :n_action] = True

        sector_masks[i] = s['sector_mask'].long()
        target_integrals[i] = s['target_integral'].long()
        start_target_integrals[i] = s['start_target_integral'].long()
        steps_remaining[i] = float(s['steps_remaining'])
        labels[i] = s['label'].long()
        _vl = s.get('valid_label_idxs')
        if _vl is None or len(_vl) == 0:
            # no recorded candidate set -> fall back to the single label, so the
            # set loss reduces to plain cross-entropy on that row instead of
            # taking log(0) over an empty set.
            label_mask[i, labels[i]] = True
        else:
            _v = _vl.long()
            # Guard against an index beyond this batch's padding width: the
            # candidate indices refer to the sample's own action list, so they
            # are < n_action <= max_action, but clamp rather than trust it.
            label_mask[i, _v[_v < max_action]] = True

    return {
        'expr_integrals': expr_integrals,
        'expr_coeffs': expr_coeffs,
        'expr_mask': expr_mask,
        'sub_keys': sub_keys,
        'sub_repl_ints': sub_repl_ints,
        'sub_repl_coeffs': sub_repl_coeffs,
        'sub_repl_mask': sub_repl_mask,
        'sub_mask': sub_mask,
        'action_ibp_ops': action_ibp_ops,
        'action_deltas': action_deltas,
        'action_mask': action_mask,
        'sector_mask': sector_masks,
        'target_integral': target_integrals,
        'start_target_integral': start_target_integrals,
        'steps_remaining': steps_remaining,
        'labels': labels,
        'label_mask': label_mask,
    }


def model_forward(model, batch):
    """Call model with the standard positional args. Both `full` and `nosubs`
    variants accept the same 13-arg signature; `nosubs` ignores the sub_*
    tensors internally."""
    kw = {}
    # EXP 1: only pass T to a model that asks for it, so the `full` variant and
    # any un-augmented `nosubs` checkpoint keep their exact 13-arg call.
    m = model.module if hasattr(model, 'module') else model
    if getattr(m, 'use_start_target', False):
        kw['start_target_integral'] = batch['start_target_integral']
    return model(
        batch['expr_integrals'], batch['expr_coeffs'], batch['expr_mask'],
        batch['sub_keys'], batch['sub_repl_ints'], batch['sub_repl_coeffs'], batch['sub_repl_mask'], batch['sub_mask'],
        batch['action_ibp_ops'], batch['action_deltas'], batch['action_mask'],
        batch['sector_mask'], batch['target_integral'], **kw,
    )


def _ddp_synced_batches(dataloader, device, max_iters=None):
    """Yield batches, stopping ALL ranks the moment ANY rank runs dry.

    WHY. DDP needs every rank to do the same number of backward passes or the
    NCCL all-reduce queues drift and the job hangs. `max_iters` guarantees that
    when the per-rank batch count is known up front -- but under token budgeting
    batches are variable, so it cannot be computed in advance (that is what the
    SystemExit at the shards/token_budget guard used to forbid).

    This gets the same guarantee dynamically: before each step every rank votes
    on whether it still has data and the MIN is taken, so ranks stop together on
    the first exhaustion. One 1-element all-reduce per step, against ~400 ms of
    compute -- unmeasurable -- and it costs at most a partial final batch, the
    same thing the static MIN cap discards.

    Outside DDP this is a plain iterator: single-process runs are unchanged.
    """
    ddp_on = dist.is_available() and dist.is_initialized()
    if not ddp_on:
        for i, b in enumerate(dataloader):
            if max_iters is not None and i >= max_iters:
                return
            yield i, b
        return

    it = iter(dataloader)
    i = -1
    while True:
        i += 1
        if max_iters is not None and i >= max_iters:
            return
        try:
            b = next(it)
        except StopIteration:
            b = None
        # MIN over ranks: a single 0 stops everyone, including the ranks that
        # still had a batch in hand (they simply discard it).
        flag = torch.tensor(1 if b is not None else 0,
                            device=device, dtype=torch.long)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if int(flag.item()) == 0:
            return
        yield i, b


def train_epoch(model, dataloader, optimizer, device, epoch, log_every=1, total_batches=None,
                max_iters=None):
    """Run one training epoch. Returns RAW sums (not averages) so the caller
    can compute correct sample-weighted means across DDP ranks."""
    model.train()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_top20 = 0
    total_samples = 0

    denom = f"/{total_batches}" if total_batches else ""
    t_prev = time.time()

    # Deterministic iteration cap — every DDP rank does EXACTLY max_iters
    # backward passes, so NCCL allreduce queues stay aligned across ranks.
    # This replaces the previous Join-based approach, which was probabilistic
    # at scale (raced after 56K iterations / 2.5h wall time). When max_iters is
    # None under DDP (token budgeting), _ddp_synced_batches keeps the ranks
    # aligned by voting instead.
    for batch_idx, batch in _ddp_synced_batches(dataloader, device, max_iters):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        logits, _ = model_forward(model, batch)

        loss = _action_loss(logits, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = batch['labels'].size(0)
        total_loss += loss.item() * bs
        total_top1 += (logits.argmax(-1) == batch['labels']).sum().item()
        _, top5 = logits.topk(min(5, logits.size(1)), dim=-1)
        total_top5 += (top5 == batch['labels'].unsqueeze(1)).any(1).sum().item()
        # top20: the beam expands the top-K actions per state, so "is the truth
        # action inside the expanded set" is what actually gates the search.
        _, top20 = logits.topk(min(20, logits.size(1)), dim=-1)
        total_top20 += (top20 == batch['labels'].unsqueeze(1)).any(1).sum().item()
        total_samples += bs

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            now = time.time()
            dt_ms = (now - t_prev) * 1000 / log_every
            t_prev = now
            print(f"  E{epoch} b{batch_idx+1}{denom} bs={bs} loss={loss.item():.4f} "
                  f"top1={total_top1/total_samples:.3f} {dt_ms:.0f}ms/batch", flush=True)

    return {
        'loss_sum': total_loss,
        'top1_sum': total_top1,
        'top5_sum': total_top5,
        'top20_sum': total_top20,
        'n_samples': total_samples,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, max_iters=None):
    """Run validation. Returns RAW sums for sample-weighted aggregation."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_top20 = 0
    total_samples = 0
    total_anyhit = 0        # top-1 lands on ANY correct action
    total_anyhit20 = 0      # ANY correct action inside the top-20
    # COVERAGE@20 -- what FRACTION of the expanded top-K are correct, not just
    # whether one is. anyhit20 is binary and saturates at 0.995-0.999, so it
    # cannot distinguish a model that squeezes one correct action in from one
    # that ranks the WHOLE valid set above the 500-3500 wrong ones. Measured on
    # the beam, soft_ep10 gets 0.52-0.88 where anyhit20 only guarantees
    # 1/20 = 0.05, and per parent lib_children ~= min(avail, 19).
    #
    # This is the quantity the BEAM actually consumes: each parent contributes
    # its top-K children, so coverage@20 IS the expected number of correct
    # children per expansion. Normalised by min(K, |C|) so a state with only 3
    # correct actions can still score 1.0 rather than being capped at 3/20.
    total_cov20 = 0.0
    # CONFIDENCE metrics. Everything above is rank-based; none of it answers
    # "is the model CONFIDENT that some correct action is correct". Under a bce
    # objective that is the question -- the model is asked to put ~1 on every
    # valid action and ~0 on the rest, and a rank metric cannot see whether it
    # did. Read from the SAME activation the model exposes to the beam.
    total_conf_best_cor = 0.0    # max sigma over CORRECT   -> should rise to 1
    total_conf_mean_cor = 0.0    # mean sigma over CORRECT  -> "confident on ALL valid"
    total_conf_max_inc = 0.0     # max sigma over INCORRECT -> should fall to 0
    total_conf_margin = 0.0      # best correct minus worst incorrect
    total_nconf_cor = 0.0        # count of CORRECT   with sigma > 0.5
    total_nconf_inc = 0.0        # count of INCORRECT with sigma > 0.5
    # RATE form of "confident on ANY correct action". The mean of max-sigma can
    # hide its distribution: 0.70 could be 0.70 everywhere, or 1.0 on 70% of
    # states and 0.0 on the rest. These count STATES where at least one correct
    # action clears the bar, which is the yes/no question the beam faces.
    total_any50 = 0.0
    total_any90 = 0.0

    _m = model.module if hasattr(model, 'module') else model
    _use_sigmoid = getattr(_m, 'score_activation', 'softmax') == 'sigmoid'

    for batch_idx, batch in _ddp_synced_batches(dataloader, device, max_iters):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, _ = model_forward(model, batch)
        loss = _action_loss(logits, batch)
        bs = batch['labels'].size(0)

        # --- confidence ---
        _act = batch['action_mask']
        _cor = batch['label_mask'] & _act
        _inc = (~batch['label_mask']) & _act
        # Padding carries -inf; sigmoid/softmax both send it to 0, and we mask
        # anyway, so no special handling is needed beyond avoiding empty sets.
        _sc = (torch.sigmoid(logits) if _use_sigmoid
               else torch.softmax(logits, dim=-1))
        _neg1 = torch.full_like(_sc, -1.0)
        _best_cor = torch.where(_cor, _sc, _neg1).max(dim=1).values
        _max_inc = torch.where(_inc, _sc, _neg1).max(dim=1).values
        _has_cor = _cor.any(dim=1)
        _has_inc = _inc.any(dim=1)
        _ncor = _cor.sum(dim=1).clamp(min=1)
        _mean_cor = (_sc * _cor).sum(dim=1) / _ncor
        total_conf_best_cor += _best_cor[_has_cor].sum().item()
        total_conf_mean_cor += _mean_cor[_has_cor].sum().item()
        total_conf_max_inc += _max_inc[_has_inc].sum().item()
        total_conf_margin += (_best_cor - _max_inc)[_has_cor & _has_inc].sum().item()
        total_nconf_cor += ((_sc > 0.5) & _cor).sum(dim=1).float().sum().item()
        total_nconf_inc += ((_sc > 0.5) & _inc).sum(dim=1).float().sum().item()
        total_any50 += (_best_cor[_has_cor] > 0.5).float().sum().item()
        total_any90 += (_best_cor[_has_cor] > 0.9).float().sum().item()
        # THE METRIC THAT MATCHES THE SEARCH. top1/top5/top20 below all score
        # against ONE designated action out of ~18 equally correct ones, so they
        # measure conformity to a convention. What the beam needs is simply that
        # the chosen action reduces -- any candidate does. Reported alongside
        # rather than instead, so the numbers stay comparable to earlier runs.
        total_anyhit += batch['label_mask'].gather(
            1, logits.argmax(-1, keepdim=True)).sum().item()
        # ANYHIT20 -- the metric that matches DEPLOYMENT exactly. The beam
        # expands the top-K=20 actions per state and any correct one lets the
        # search proceed, so what gates it is "is at least one correct action
        # inside the expanded set". anyhit above is the K=1 special case, and
        # top20 is the other partial view (right K, but scored against the one
        # arbitrarily-designated action). This is the union of both leniencies.
        _k = min(20, logits.size(1))
        _, _t20 = logits.topk(_k, dim=-1)
        _hit20 = batch['label_mask'].gather(1, _t20)
        total_anyhit20 += _hit20.any(1).sum().item()
        _ncor = batch['label_mask'].sum(1).clamp(min=1)
        total_cov20 += (_hit20.sum(1).float()
                        / torch.minimum(_ncor,
                                        torch.full_like(_ncor, _k)).float()
                        ).sum().item()
        total_loss += loss.item() * bs
        total_top1 += (logits.argmax(-1) == batch['labels']).sum().item()
        _, top5 = logits.topk(min(5, logits.size(1)), dim=-1)
        total_top5 += (top5 == batch['labels'].unsqueeze(1)).any(1).sum().item()
        # top20: the beam expands the top-K actions per state, so "is the truth
        # action inside the expanded set" is what actually gates the search.
        _, top20 = logits.topk(min(20, logits.size(1)), dim=-1)
        total_top20 += (top20 == batch['labels'].unsqueeze(1)).any(1).sum().item()
        total_samples += bs

    return {
        'loss_sum': total_loss,
        'top1_sum': total_top1,
        'top5_sum': total_top5,
        'top20_sum': total_top20,
        'anyhit_sum': total_anyhit,
        'anyhit20_sum': total_anyhit20,
        'cov20_sum': total_cov20,
        'conf_best_cor_sum': total_conf_best_cor,
        'conf_mean_cor_sum': total_conf_mean_cor,
        'conf_max_inc_sum': total_conf_max_inc,
        'conf_margin_sum': total_conf_margin,
        'nconf_cor_sum': total_nconf_cor,
        'nconf_inc_sum': total_nconf_inc,
        'any50_sum': total_any50,
        'any90_sum': total_any90,
        'n_samples': total_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', type=str, required=True,
                        help='Path to topology_input/<family>/ directory '
                             '(determines n_indices, n_denominators, n_actions)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Single-pack mode: directory with train.pt/val.pt. '
                             'Mutually exclusive with --shards_dir.')
    parser.add_argument('--shards_dir', type=str, default=None,
                        help='Sharded mode: directory containing shard_*/train.pt etc. '
                             'Uses ShardedIBPDataset (memory-bounded streaming).')
    parser.add_argument('--n_val_shards', type=str, default='50',
                        help='How many shards to use for validation each epoch '
                             '(int, or "all"). Ignored in --data_dir mode.')
    parser.add_argument('--max_train_shards', type=int, default=0,
                        help='Cap the number of train shards (0 = use all). For smoke tests.')
    parser.add_argument('--buffer_shards', type=int, default=4,
                        help='Number of shards held in the train shuffle buffer '
                             '(per DataLoader worker). Larger = better mixing, more RAM.')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--select_on', type=str, default='val_loss',
                        choices=['val_loss', 'val_top20', 'val_top1',
                                 'val_anyhit', 'val_anyhit20',
                                 'conf_margin', 'conf_max_incorrect',
                                 'extra_top20', 'extra_top1',
                                 'dots_top20', 'dots_top1'],
                        help='Metric that decides best_model.pt. Default '
                             'val_loss. NOTE: which metric to select on is '
                             'NOT settled, and there is measured evidence '
                             'both ways. (a) Against val_loss: cross-entropy '
                             'is dominated by a shrinking set of '
                             'confidently-WRONG samples (measured: wrong-sample '
                             'loss 5.43->7.40 while top1/top5/top20 all rose), '
                             'so it can select a checkpoint better calibrated '
                             'on hopeless states rather than one that RANKS '
                             'the truth action into the top-K the beam '
                             'expands. (b) For val_loss: on an earlier '
                             'campaign val_top20 picked a checkpoint that then '
                             'FAILED the beam campaign while the val_loss '
                             'minimum succeeded. The arbiter is beam solve '
                             'behaviour on the 125 test integrals, not any '
                             'val-set number -- pick per run and record which. '
                             'The gravity3L p=101 from-scratch run uses '
                             'val_loss (see training/TRAIN_FROM_SCRATCH.md).')
    parser.add_argument('--backbone_lr', type=float, default=0.0,
                        help='If >0, train the BACKBONE (expr_enc, sector_enc, '
                             'action_enc) at this lower LR while the head '
                             '(state_combine, scorer) uses --lr. Middle ground '
                             'between full fine-tuning (backbone drifts, '
                             'forgetting) and --freeze (no adaptation at all). '
                             'Both groups still follow the same cosine decay.')
    parser.add_argument('--use_value_head', action='store_true',
                        help='EXP 2: add a head predicting steps-to-go. The '
                             'beam has no cost-to-go signal; this supplies one. '
                             'Label is free (trajectory length - step index).')
    parser.add_argument('--value_weight', type=float, default=1.0,
                        help='weight of the value regression in the total loss')
    parser.add_argument('--use_start_target', action='store_true',
                        help='EXP 1: feed the START target T as a 4th state '
                             'channel. Widens state_combine, so a v7 checkpoint '
                             'must be converted first (see '
                             'convert_v7_to_start_target.py).')
    parser.add_argument('--freeze', type=str, default='',
                        help='Comma-separated top-level module names to FREEZE '
                             '(requires_grad=False), e.g. '
                             '"expr_enc,sector_enc,action_enc" for head-only '
                             'fine-tuning. Freezing makes catastrophic '
                             'forgetting impossible in those modules by '
                             'construction (vs merely penalised by L2-SP) and '
                             'cuts trainable capacity, which is the binding '
                             'constraint on a small dots corpus.')
    parser.add_argument('--extra_val', type=str, default=None,
                        help='Optional SECOND held-out packed .pt evaluated '
                             'each epoch, logged as extra_* columns. Its '
                             'contents are whatever file you point at — the '
                             'columns are named "extra" precisely so they '
                             'never imply a distribution. The MAIN val set is '
                             'always <data_dir>/val.pt and is logged as val_*.')
    parser.add_argument('--dots_val', type=str, default=None,
                        help='DEPRECATED alias for --extra_val. The old name '
                             'caused a real misreading: runs pointed it at a '
                             'REPLAY holdout while the dots holdout sat in '
                             'val.pt, so "dots_*" columns held replay numbers '
                             'and vice-versa. Use --extra_val.')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Optional path to write a per-epoch metrics TSV (epoch, train_loss, ...).')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'plateau'],
                        help='LR schedule. "cosine": CosineAnnealingLR with '
                             'T_max=--epochs, eta_min=lr/10 (the original '
                             'behaviour). "plateau": ReduceLROnPlateau on '
                             'val_loss -- cuts LR by --plateau_factor after '
                             '--plateau_patience epochs without a val_loss '
                             'improvement. Use plateau when the convergence '
                             'horizon is unknown: cosine with a large T_max '
                             'spends its first quarter at ~peak LR, so if the '
                             'model plateaus early the anneal arrives far too '
                             'late to help.')
    parser.add_argument('--plateau_factor', type=float, default=0.3,
                        help='--lr_schedule plateau: multiply LR by this on each cut.')
    parser.add_argument('--plateau_patience', type=int, default=5,
                        help='--lr_schedule plateau: epochs without improvement '
                             'before cutting. MUST exceed the val_loss '
                             'oscillation period or it fires on noise '
                             '(observed on gravity3L p101: ~3-epoch dip-and-'
                             'recover swings, so 5 is the floor).')
    parser.add_argument('--plateau_threshold', type=float, default=1e-3,
                        help='--lr_schedule plateau: relative improvement below '
                             'this counts as no improvement.')
    parser.add_argument('--plateau_min_lr', type=float, default=1e-6,
                        help='--lr_schedule plateau: floor for the LR.')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--token_budget', type=int, default=0,
                        help='If >0 (--data_dir mode only): use size-bucketed '
                             'batches with len(batch)*max_actions <= budget '
                             '(and len(batch) <= batch_size). Full action '
                             'lists, no OOM from fat samples.')
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_expr_layers', type=int, default=2)
    parser.add_argument('--n_cross_layers', type=int, default=2)
    parser.add_argument('--n_subs_layers', type=int, default=2)
    parser.add_argument('--model_variant', type=str, default='nosubs',
                        choices=tuple(MODEL_CLASSES.keys()),
                        help='full: original IBPActionClassifier (with subs encoder); '
                             'nosubs: IBPActionClassifierNoSubs (subs path removed, recommended). '
                             'Checkpoints from different variants are NOT interchangeable; use a '
                             'distinct --output_dir per variant.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prime', type=int, default=1009)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint to resume from.')
    parser.add_argument('--auto_resume', action='store_true',
                        help='If <output_dir>/last.pt exists, resume from it. '
                             'Overrides --resume when both are set and last.pt exists.')
    parser.add_argument('--checkpoint_every', type=int, default=5,
                        help='Also save a numbered checkpoint every N epochs (in addition to last.pt).')
    parser.add_argument('--restart_lr', type=float, default=0.0,
                        help='On resume, FORCE the LR to this value and reset the '
                             'scheduler state. Without it a resume restores the '
                             'decayed LR and the scheduler\'s accumulated "best", '
                             'so a run that has already annealed cannot be revived '
                             '-- and if --select_on changed, the restored "best" '
                             'refers to a different metric entirely.')
    parser.add_argument('--log_every', type=int, default=1,
                        help='Print per-batch metrics every N batches (1 = every batch).')
    args = parser.parse_args()
    if args.extra_val is None and args.dots_val is not None:
        args.extra_val = args.dots_val
        print('NOTE: --dots_val is deprecated; use --extra_val '
              '(columns are logged as extra_*).', flush=True)

    if (args.data_dir is None) == (args.shards_dir is None):
        parser.error('Exactly one of --data_dir or --shards_dir is required.')

    # DDP detection: torchrun sets LOCAL_RANK/RANK/WORLD_SIZE env vars.
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    ddp_enabled = local_rank >= 0
    # Global rank — distinct from local_rank on multi-node (one local_rank==0
    # per node, but exactly one global_rank==0 across the whole job).
    global_rank = int(os.environ.get('RANK', 0))

    # Build manifest on global rank 0 BEFORE CUDA is touched.
    # multiprocessing.Pool uses fork, which is safe pre-CUDA-init.
    # ~2-3 min one-time; cached to JSON for future runs.
    if args.shards_dir and (not ddp_enabled or global_rank == 0):
        manifest_path = Path(args.shards_dir) / 'manifest.json'
        if not manifest_path.is_file():
            print(f"[rank{global_rank}] Building manifest at {manifest_path} "
                  f"(one-time, ~2-3 min)...", flush=True)
            from sharded_dataset import build_manifest as _bm
            _bm(args.shards_dir)
            print(f"[rank{global_rank}] Manifest built.", flush=True)

    if ddp_enabled:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
        # CRITICAL: synchronize all ranks here. With --standalone rendezvous,
        # init_process_group returns as soon as each rank registers with the
        # store (microseconds) — it does NOT wait for other ranks. NCCL
        # communicator is created lazily on first collective op. So if we
        # don't barrier here, ranks 1/2 race ahead and try to read the manifest
        # before rank 0 has finished building it.
        dist.barrier(device_ids=[local_rank])
    else:
        rank = 0
        world_size = 1
        device = args.device
    is_main = (rank == 0)

    def log(msg):
        if is_main:
            print(msg, flush=True)

    # Load topology to determine model dimensions.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from sailir.topology import Topology
    topology = Topology.from_dir(args.topology)
    n_indices = topology.n_indices
    n_denominators = topology.n_denominators
    n_actions = topology.n_actions

    log("=" * 70)
    log("SAILIR IBP action classifier — training")
    if ddp_enabled:
        log(f"  - DDP: rank {rank}/{world_size} on cuda:{local_rank}")
    log("=" * 70)
    for k, v in vars(args).items():
        log(f"  {k}: {v}")

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if args.data_dir else None

    collate = make_collate_fn(
        n_indices=n_indices, n_denominators=n_denominators,
    )

    if args.shards_dir is not None:
        # Sharded streaming mode — memory-bounded.
        from sharded_dataset import (
            ShardedIBPDataset, discover_shards, estimate_samples_per_epoch,
        )

        log(f"\nDiscovering shards in {args.shards_dir}...")
        train_shards = discover_shards(args.shards_dir, 'train')
        val_shards_all = discover_shards(args.shards_dir, 'val')
        log(f"  train shards: {len(train_shards)}")
        log(f"  val shards available: {len(val_shards_all)}")
        if not train_shards:
            raise RuntimeError(f"No train.pt shards found under {args.shards_dir}")

        if args.max_train_shards > 0:
            train_shards = train_shards[: args.max_train_shards]
            log(f"  --max_train_shards: capped train to {len(train_shards)} shards")

        if args.n_val_shards == 'all':
            val_shards = val_shards_all
        else:
            val_shards = val_shards_all[: int(args.n_val_shards)]
        log(f"  val shards selected: {len(val_shards)}")

        # In DDP mode, truncate to a multiple of world_size so every rank
        # iterates the same number of shards → matching batch counts at
        # all-reduce time (DDP would otherwise hang on the trailing batches).
        if ddp_enabled and world_size > 1:
            for name, shards_list in (('train', train_shards), ('val', val_shards)):
                n_keep = (len(shards_list) // world_size) * world_size
                if n_keep < len(shards_list):
                    log(f"  {name}: truncating to {n_keep} shards (world_size={world_size}; "
                        f"dropping {len(shards_list)-n_keep})")
                shards_list[:] = shards_list[:n_keep]

        # Manifest is guaranteed to exist here (built pre-DDP by rank 0).
        manifest_path = Path(args.shards_dir) / 'manifest.json'
        with open(manifest_path) as f:
            manifest = json.load(f)
        log(f"  manifest: {manifest_path}")

        n_train_samples = sum(manifest['splits']['train'].get(str(p), 0) for p in train_shards)
        n_val_samples = sum(manifest['splits']['val'].get(str(p), 0) for p in val_shards)
        log(f"  train samples/epoch (global): {n_train_samples:,}")
        log(f"  val   samples/epoch (global): {n_val_samples:,}")

        # Per-rank exact batch counts (mirror the rank-stride in ShardedIBPDataset.__iter__).
        rank_train_paths = train_shards[rank::world_size]
        rank_val_paths = val_shards[rank::world_size]
        local_train_samples = sum(manifest['splits']['train'][str(p)] for p in rank_train_paths)
        local_val_samples = sum(manifest['splits']['val'][str(p)] for p in rank_val_paths)
        local_train_batches = local_train_samples // args.batch_size
        local_val_batches = local_val_samples // args.batch_size

        # Global iteration cap = MIN across ranks. Every rank does EXACTLY this
        # many backward passes per epoch → NCCL allreduce queues stay aligned by
        # construction, no possibility of train→val boundary desync.
        if ddp_enabled:
            t = torch.tensor(local_train_batches, device=device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            train_iter_cap = int(t.item())
            t = torch.tensor(local_val_batches, device=device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            val_iter_cap = int(t.item())
        else:
            train_iter_cap = local_train_batches
            val_iter_cap = local_val_batches

        # The cap above assumes a FIXED batch size (samples // batch_size). With
        # the token budget on, batches are variable and far more numerous -- a
        # sample with >32k actions rides alone -- so that number is a large
        # UNDER-count and would silently truncate the epoch to a few percent of
        # the corpus, with no error. The cap exists only to keep DDP ranks doing
        # equal numbers of backward passes; single-process runs do not need it.
        if args.token_budget:
            # DDP + token budgeting used to be refused here: the per-rank batch
            # count is not samples//batch_size when batches are variable, so no
            # cap can be computed up front, and unequal backward-pass counts
            # hang NCCL. _ddp_synced_batches now supplies that guarantee
            # dynamically (ranks vote each step and stop together), so the cap
            # is no longer needed and the combination is allowed.
            #
            # Do NOT "fix" this by setting --token_budget 0 instead: that path
            # bypasses ShardedIBPDataset._emit, which is what sorts each window
            # by action count. Measured on the corrupt cull corpus, sorted
            # batching pads 1.009x vs 1.85x for unsorted -- turning the budget
            # off would nearly double the model's work and cancel the DDP win.
            train_iter_cap = None
            val_iter_cap = None
            log("  iteration cap: DISABLED (token budget -> variable batch "
                "sizes; a fixed cap would truncate the epoch)")
        else:
            log(f"  iteration cap: train={train_iter_cap}/rank, val={val_iter_cap}/rank "
                f"(this rank had: {local_train_batches} train, {local_val_batches} val)")

        # Give the shards path the SAME padded-memory bound the --data_dir path
        # gets from TokenBudgetBatchSampler. Without it this path builds flat
        # batch_size batches, which at ~12,700 actions/sample is ~813k tokens --
        # 12.4x the budget -- and OOMs a 15.6 GB GPU in the action encoder.
        # The dataset then yields LISTS, so the DataLoader below must use
        # batch_size=None (automatic batching off) and let collate see the list.
        _tb = args.token_budget if args.token_budget else 0
        train_dataset = ShardedIBPDataset(
            train_shards, shuffle=True, buffer_shards=args.buffer_shards, seed=args.seed,
            rank=rank, world_size=world_size,
            token_budget=_tb, max_batch=args.batch_size,
        )
        val_dataset = ShardedIBPDataset(
            val_shards, shuffle=False, buffer_shards=1, seed=args.seed,
            rank=rank, world_size=world_size,
            token_budget=_tb, max_batch=args.batch_size,
        )
        # IterableDataset → shuffle=False at DataLoader; shuffling is handled
        # inside the dataset. persistent_workers=False so each iter(loader)
        # re-pickles the dataset to fresh workers, carrying the updated
        # _epoch from set_epoch() — persistent workers would freeze at _epoch=0.
        # batch_size=None when the dataset is already emitting token-budgeted
        # LISTS: automatic batching off, so collate_fn receives the list as-is.
        _bs = None if _tb else args.batch_size
        train_loader = DataLoader(
            train_dataset, batch_size=_bs, shuffle=False,
            collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
            persistent_workers=False,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=_bs, shuffle=False,
            collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
            persistent_workers=False,
        )
        # Per-rank batch count for the progress denominator string.
        train_sampler = None
        approx_train_batches = train_iter_cap
        approx_val_batches = val_iter_cap
    else:
        log("\nLoading packed tensors...")
        t0 = time.time()
        train_data = torch.load(data_dir / 'train.pt', weights_only=False)
        val_data = torch.load(data_dir / 'val.pt', weights_only=False)
        log(f"Loaded in {time.time() - t0:.1f}s")

        assert 'target_integrals' in train_data, "Missing target_integrals!"
        log(f"  target_integrals shape: {train_data['target_integrals'].shape}")
        # subs are OPTIONAL now: a corpus packed with include_subs=False has
        # none, because the nosubs model deletes them on entry to forward().
        if 'subs_raw' in train_data:
            log(f"  subs_raw count: {len(train_data['subs_raw'])}")
        else:
            log("  subs: ABSENT (corpus packed without substitutions; "
                "requires --model_variant nosubs)")

        train_dataset = PackedDatasetV5(train_data)
        val_dataset = PackedDatasetV5(val_data)
        log(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        if args.token_budget > 0:
            train_sampler = TokenBudgetBatchSampler(
                train_data['num_valid_actions'].tolist(), args.token_budget,
                args.batch_size, seed=args.seed)
            val_sampler = TokenBudgetBatchSampler(
                val_data['num_valid_actions'].tolist(), args.token_budget,
                args.batch_size, seed=args.seed)
            log(f"Token-budget batching: budget={args.token_budget}, "
                f"train batches={len(train_sampler)}, val batches={len(val_sampler)}")
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                      collate_fn=collate, num_workers=args.num_workers,
                                      pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                                    collate_fn=collate, num_workers=args.num_workers,
                                    pin_memory=True)
        else:
            train_sampler = None
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=collate, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=collate, num_workers=args.num_workers, pin_memory=True)
        approx_train_batches = len(train_loader)
        approx_val_batches = len(val_loader)
        # No cap needed for map-style dataset (DistributedSampler would give exact
        # parity if DDP were enabled; we don't use DDP in --data_dir mode here).
        train_iter_cap = None
        val_iter_cap = None

        log(f"Train batches: {approx_train_batches}, Val batches: {approx_val_batches}")

    # Extra dots-only held-out eval (deployment distribution).
    dots_loader = None
    if args.extra_val and os.path.exists(args.extra_val):
        _dd = torch.load(args.extra_val, map_location='cpu', weights_only=False)
        _dds = PackedDatasetV5(_dd)
        _dsamp = TokenBudgetBatchSampler(
            _dd['num_valid_actions'].tolist(), args.token_budget or 65536,
            args.batch_size, seed=args.seed) if args.token_budget > 0 else None
        dots_loader = (DataLoader(_dds, batch_sampler=_dsamp, collate_fn=collate,
                                  num_workers=args.num_workers, pin_memory=True)
                       if _dsamp is not None else
                       DataLoader(_dds, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=collate, num_workers=args.num_workers,
                                  pin_memory=True))
        log(f"Extra held-out eval [{os.path.basename(args.extra_val)}]: "
                f"{len(_dds)} samples -> logged as extra_* columns")

    from sailir import ibp_env
    ibp_env.init_from_topology(topology)
    ibp_env.set_prime(args.prime)

    # A corpus packed with include_subs=False has NO substitution data. The
    # collate would quietly hand the full model all-zero substitution tensors
    # and it would train on them without complaint, so refuse the combination
    # outright rather than produce a silently-degraded model.
    # 'eqact' subclasses the nosubs model and deletes the sub_* arguments in
    # forward() exactly as it does, so it needs no packed substitutions either;
    # its action equations are resolved from the on-disk store by
    # training/eqact_dataset.py. Listing it here rather than aliasing it to
    # 'nosubs': the variant string is written into the checkpoint args, and
    # onestep_worker_v9 constructs the model class from that field, so a lie
    # here would build the wrong class at inference.
    if args.model_variant not in ('nosubs', 'eqact'):
        _probe = None
        if args.data_dir:
            _probe = train_data
        elif train_shards:
            _probe = torch.load(train_shards[0], map_location='cpu',
                                weights_only=False)
        if _probe is not None and 'subs_raw' not in _probe:
            raise SystemExit(
                f"--model_variant {args.model_variant} needs substitution data, "
                f"but this corpus was packed without it (include_subs=False). "
                f"Repack from the recordings, or use --model_variant nosubs.")
        del _probe

    model_cls = MODEL_CLASSES[args.model_variant]
    # A bce-trained model MUST expose sigmoid scores, not softmax: the beam
    # consumes forward()'s second value as action_prob, and softmax divides by
    # sum_j exp(z_j) over the action set -- reintroducing at inference the very
    # action-count dependence the bce objective removes during training.
    # Recorded into args (hence into every checkpoint) so the beam can rebuild
    # the model the same way.
    args.score_activation = 'sigmoid' if _BCE_LOSS else 'softmax'
    if _BCE_LOSS and args.model_variant != 'nosubs':
        raise SystemExit(
            f"SAILIR_LOSS=bce needs score_activation support, which is "
            f"implemented on 'nosubs' only; got --model_variant "
            f"{args.model_variant!r}.")
    model = model_cls(
        embed_dim=args.embed_dim, n_heads=args.n_heads,
        n_expr_layers=args.n_expr_layers, n_cross_layers=args.n_cross_layers,
        n_subs_layers=args.n_subs_layers, prime=args.prime,
        n_indices=n_indices, n_denominators=n_denominators, n_ibp_ops=n_actions,
        **({'use_start_target': True} if args.use_start_target else {}),
        **({'use_value_head': True} if args.use_value_head else {}),
        **({'score_activation': 'sigmoid'} if _BCE_LOSS else {}),
    )
    model = model.to(device)
    log(f"Model variant: {args.model_variant} ({model_cls.__name__})")
    log(f"Loss: {os.environ.get('SAILIR_LOSS', 'ce')}  "
        f"score_activation: {args.score_activation}")
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.freeze:
        _want = [m.strip() for m in args.freeze.split(',') if m.strip()]
        _avail = dict(model.named_children())
        _bad = [m for m in _want if m not in _avail]
        if _bad:
            raise SystemExit(f"--freeze: unknown module(s) {_bad}; "
                             f"available: {sorted(_avail)}")
        for _m in _want:
            for _p in _avail[_m].parameters():
                _p.requires_grad = False
        _tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        log(f"FROZEN modules: {_want}")
        log(f"  trainable={_tr:,}  frozen={_fr:,}  "
            f"({100*_tr/(_tr+_fr):.1f}% trainable)")

    if ddp_enabled:
        # DDP: one process per GPU, model lives permanently on each. Only
        # gradients cross the interconnect, via NCCL all-reduce overlapped
        # with backward.
        # static_graph: REQUIRED when the model uses activation checkpointing
        # (eqact does, SAILIR_EQ_CKPT=1). Checkpointing re-runs the forward
        # during backward, so DDP's reducer never sees those parameters in the
        # first pass and aborts with "Expected to have finished reduction in the
        # prior iteration ... parameters that were not used in producing loss"
        # (observed: indices 279-286, the equation encoder). static_graph tells
        # DDP the parameter set is identical every iteration, which is true
        # here, and is cheaper than find_unused_parameters=True.
        _static = os.environ.get('SAILIR_DDP_STATIC_GRAPH') == '1'
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False, static_graph=_static)
        log(f"Using DistributedDataParallel: world_size={world_size} "
            f"static_graph={_static}")
        base_model = model.module
    elif device == 'cuda' and torch.cuda.device_count() > 1:
        # Legacy DataParallel path — known to be slow on SYS-grade interconnects
        # (separate PCIe roots). Kept for environments where it works (NVLink
        # or shared PCIe switch); torchrun → DDP is the preferred path here.
        n_gpu = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
        log(f"Using DataParallel over {n_gpu} GPUs: {names}")
        model = torch.nn.DataParallel(model)
        base_model = model.module
    else:
        base_model = model

    # Only optimise trainable params: with --freeze, passing frozen tensors to
    # AdamW would still let weight_decay act on them (decay is applied to the
    # param, not via .grad), silently drifting the "frozen" backbone.
    _BACKBONE = ('expr_enc', 'sector_enc', 'action_enc')
    if args.backbone_lr > 0:
        _kids = dict(base_model.named_children())
        _bb, _hd = [], []
        # DEDUPE BY IDENTITY. With SAILIR_EQ_TIE, eq_enc shares modules with
        # (or IS) expr_enc, so the same tensor is reachable from two children.
        # Adding it to two param groups makes AdamW raise "some parameters
        # appear in more than one parameter group"; adding it twice to ONE
        # group would double its effective learning rate just as silently.
        # Backbone wins the tie: a shared embedder is pretrained, so it belongs
        # at the fine-tuning lr. Genuinely NEW tensors are lifted back out to
        # the head lr by the _NEW rule below.
        # Backbone children FIRST and explicitly, so "backbone wins" is a
        # property of this loop rather than of dict registration order.
        _seen = set()
        for _want_bb in (True, False):
            for _n, _m in _kids.items():
                if (_n in _BACKBONE) != _want_bb:
                    continue
                _dst = _bb if _want_bb else _hd
                for q in _m.parameters():
                    if q.requires_grad and id(q) not in _seen:
                        _seen.add(id(q))
                        _dst.append(q)
        # BARE PARAMETERS. named_children() yields sub-MODULES only, so an
        # nn.Parameter declared directly on the model belongs to no child and
        # was silently dropped from BOTH groups -- it then never updates.
        # This cost 3 epochs: eqact's `eq_gate` is exactly such a scalar, so it
        # stayed pinned at 0, and because the equation branch is added as
        # `gate * branch`, every gradient into eq_enc/eq_proj was 0 too. All
        # 111 eq_* tensors were bit-identical between epoch 1 and epoch 3 while
        # all 186 others moved: the run was plain `nosubs` wearing eqact's name.
        _bare = [q for _, q in base_model.named_parameters(recurse=False)
                 if q.requires_grad]
        _hd.extend(_bare)
        # FRESHLY-INITIALISED parameters belong at the HEAD lr, not the
        # backbone's fine-tuning lr, wherever they physically live. The
        # coefficient lookup table sits inside expr_enc (a _BACKBONE child) but
        # also inside eq_enc (not one), so without this it would train at 1e-5
        # on one copy and 1e-4 on the other -- an asymmetry with no meaning.
        _NEW = ('coeff_table',)
        _name_of = {id(q): n for n, q in base_model.named_parameters()}
        _moved = [q for q in _bb
                  if any(t in _name_of.get(id(q), '') for t in _NEW)]
        if _moved:
            _mids = {id(q) for q in _moved}
            _bb = [q for q in _bb if id(q) not in _mids]
            _hd.extend(_moved)
            log(f"  new-parameter routing: {len(_moved)} tensor(s) matching "
                f"{_NEW} moved from backbone to head lr")
        optimizer = torch.optim.AdamW(
            [{'params': _bb, 'lr': args.backbone_lr},
             {'params': _hd, 'lr': args.lr}],
            lr=args.lr, weight_decay=args.weight_decay)
        # Nothing trainable may go unoptimised. Silence here is what hid the
        # bug -- the only symptom was a parameter count one short of the model.
        _grouped = sum(q.numel() for q in _bb) + sum(q.numel() for q in _hd)
        _trainable = sum(q.numel() for q in base_model.parameters()
                         if q.requires_grad)
        if _grouped != _trainable:
            raise SystemExit(
                f'optimizer groups cover {_grouped:,} of {_trainable:,} '
                f'trainable params -- {_trainable - _grouped:,} would never '
                f'update. Refusing to train.')
        log(f"Discriminative LR: backbone({','.join(_BACKBONE)})="
            f"{args.backbone_lr:.2e} over {sum(q.numel() for q in _bb):,} params"
            f" | head={args.lr:.2e} over {sum(q.numel() for q in _hd):,} params"
            f" | bare-on-model: {len(_bare)} tensor(s) -> head")
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.plateau_factor,
            patience=args.plateau_patience, threshold=args.plateau_threshold,
            threshold_mode='rel', min_lr=args.plateau_min_lr)
        log(f"LR schedule: ReduceLROnPlateau({args.select_on}) factor={args.plateau_factor} "
            f"patience={args.plateau_patience} threshold={args.plateau_threshold} "
            f"min_lr={args.plateau_min_lr}")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr/10)
        log(f"LR schedule: CosineAnnealingLR T_max={args.epochs} eta_min={args.lr/10:.2e}")

    def _full_state(epoch, val_metrics, train_metrics=None):
        # Persist enough state that a resumed run continues the cosine LR
        # schedule and best-val tracking without drift.
        return {
            'epoch': epoch,
            'model_state_dict': base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            # Record which schedule produced that state so a resume can refuse
            # to load a CosineAnnealingLR state into a ReduceLROnPlateau (their
            # state_dicts share no keys; load_state_dict would silently inject
            # junk attributes rather than error).
            'lr_schedule': args.lr_schedule,
            'val_metrics': val_metrics,
            'train_metrics': train_metrics,
            'best_val_loss': best_val_loss,
            'args': vars(args),
        }

    start_epoch, best_val_loss = 1, float('inf')
    resume_path = None
    if args.auto_resume and (output_dir / 'last.pt').is_file():
        resume_path = output_dir / 'last.pt'
        log(f"--auto_resume: found {resume_path}")
    elif args.resume:
        resume_path = Path(args.resume)
    if resume_path is not None:
        # map_location keeps each rank's parameters on its own device.
        ckpt = torch.load(resume_path, weights_only=False, map_location=device)
        _sd = ckpt['model_state_dict']
        # WIDENING FOR use_start_target. The extra state channel makes
        # state_combine's first Linear expect embed_dim*4 inputs where a
        # checkpoint trained without it has embed_dim*3, so a plain load fails
        # on shape. The channel is APPENDED last (classifier_nosubs.py:134,148),
        # so the checkpoint's columns map onto the leading block and zeroing the
        # remainder makes the new channel contribute exactly 0 -- the model
        # reproduces the checkpoint and the channel must earn its way in.
        # start_target_enc/start_proj are then legitimately absent; they feed
        # ONLY that zeroed block, so their random init is multiplied by zero.
        # Anything else missing is still a hard error.
        if getattr(base_model, 'use_start_target', False):
            _own = base_model.state_dict()
            _k = 'state_combine.0.weight'
            if _k in _sd and _sd[_k].shape[1] < _own[_k].shape[1]:
                _wide = torch.zeros_like(_own[_k])
                _wide[:, :_sd[_k].shape[1]] = _sd[_k]
                _sd = dict(_sd); _sd[_k] = _wide
                log(f"  widened {_k} {tuple(ckpt['model_state_dict'][_k].shape)}"
                    f" -> {tuple(_own[_k].shape)}; start-target block zeroed "
                    f"-> model is bit-identical to the checkpoint")
            _missing = set(_own) - set(_sd)
            if _missing and all(('start_target_enc' in m or 'start_proj' in m)
                                for m in _missing):
                log(f"  {len(_missing)} start-target tensor(s) absent from the "
                    f"checkpoint, left at init (inert behind the zeroed block)")
                base_model.load_state_dict(_sd, strict=False)
            else:
                base_model.load_state_dict(_sd)
        else:
            base_model.load_state_dict(_sd)
        # With --freeze the optimizer holds only the trainable subset, so a
        # checkpoint's optimizer state (built over ALL params) has a
        # mismatched param group and cannot be loaded. Its momentum/variance
        # buffers are meaningless for a frozen subset anyway — start fresh.
        if args.freeze or args.backbone_lr > 0:
            log("  --freeze/--backbone_lr set: skipping optimizer state "
                "(param groups differ from the checkpoint's)")
        else:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        ckpt_sched = ckpt.get('lr_schedule', 'cosine')
        if ckpt_sched != args.lr_schedule:
            # Deliberate schedule change across a resume. Keep the model and
            # optimizer state, start the new schedule fresh -- for plateau that
            # means its "best" begins unset, so the first epochs after the
            # switch re-establish the baseline before any cut can fire.
            log(f"  LR schedule changed ({ckpt_sched} -> {args.lr_schedule}): "
                f"starting the new scheduler fresh, NOT loading its state")
        elif 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        elif args.lr_schedule == 'cosine':
            for _ in range(ckpt['epoch']):
                scheduler.step()
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', ckpt['val_metrics']['loss'])
        log(f"Resumed from epoch {start_epoch-1}, best_val_loss={best_val_loss:.4f}")

        # A checkpoint's best_val_loss is in the units of the --select_on it was
        # trained with. Resuming with a DIFFERENT --select_on would compare
        # incomparable numbers and could freeze best_model.pt forever, so reset.
        _ck_sel = (ckpt.get('args') or {}).get('select_on', 'val_loss')
        _ck_dir = (ckpt.get('args') or {}).get('shards_dir')
        if _ck_sel != args.select_on:
            log(f"  --select_on changed ({_ck_sel} -> {args.select_on}): "
                f"resetting best score (the stored one is in the old metric's units)")
            best_val_loss = float('inf')
        elif _ck_dir and _ck_dir != args.shards_dir:
            # Same failure, different cause: the score is only comparable
            # against the val SET it was computed on. DAgger fine-tuning
            # aggregates harder off-path rows into val, so the inherited best
            # (measured on the easier base split) can be unbeatable and would
            # freeze best_model.pt for the whole run.
            log(f"  --shards_dir changed ({_ck_dir} -> {args.shards_dir}): "
                f"resetting best score (the stored one is on a different val set)")
            best_val_loss = float('inf')

        if args.restart_lr > 0:
            # Force the LR back up and wipe the scheduler's history. Without
            # this a resumed run keeps the annealed LR (and, for plateau, a
            # 'best' accumulated under the old metric), so it cannot recover.
            for g in optimizer.param_groups:
                g['lr'] = args.restart_lr
            if args.lr_schedule == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=args.plateau_factor,
                    patience=args.plateau_patience, threshold=args.plateau_threshold,
                    threshold_mode='rel', min_lr=args.plateau_min_lr)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=args.lr/10)
            log(f"  --restart_lr: LR forced to {args.restart_lr:.2e}, scheduler reset")

    # Per-epoch metrics log — only rank 0 writes.
    log_fp = None
    if args.log_file and is_main:
        log_fp = open(args.log_file, 'a')
        if log_fp.tell() == 0:
            log_fp.write("epoch\ttrain_loss\ttrain_top1\ttrain_top5\tval_loss\tval_top1\tval_top5"
                         "\textra_loss\textra_top1\textra_top5"
                         "\tval_top20\textra_top20\tval_anyhit\tval_anyhit20"
                         "\tval_cov20\tconf_best_correct\tconf_mean_correct"
                         "\tconf_max_incorrect\tconf_margin"
                         "\tn_conf_correct\tn_conf_incorrect"
                         "\tconf_any50\tconf_any90\tlr\twall_s\n")
            log_fp.flush()

    def aggregate_metrics(raw):
        """Sum raw counts across DDP ranks then divide by global sample count.

        Correct under uneven per-rank sample counts (where mean-of-means would
        be biased). Also guarantees top5_acc >= top1_acc because we sum raw
        counts before dividing.
        """
        loss_sum = raw['loss_sum']
        top1_sum = raw['top1_sum']
        top5_sum = raw['top5_sum']
        top20_sum = raw.get('top20_sum', 0)
        # 0 for the TRAIN loop, which does not compute it (it is a validation
        # diagnostic, and the train loop's job is the loss).
        anyhit_sum = raw.get('anyhit_sum', 0)
        anyhit20_sum = raw.get('anyhit20_sum', 0)
        cov20_sum = raw.get('cov20_sum', 0.0)
        cbc = raw.get('conf_best_cor_sum', 0.0)
        cmc = raw.get('conf_mean_cor_sum', 0.0)
        cmi = raw.get('conf_max_inc_sum', 0.0)
        cmg = raw.get('conf_margin_sum', 0.0)
        ncc = raw.get('nconf_cor_sum', 0.0)
        nci = raw.get('nconf_inc_sum', 0.0)
        a50 = raw.get('any50_sum', 0.0)
        a90 = raw.get('any90_sum', 0.0)
        n = raw['n_samples']
        if ddp_enabled:
            t = torch.tensor([loss_sum, top1_sum, top5_sum, top20_sum,
                              anyhit_sum, anyhit20_sum, cov20_sum,
                              cbc, cmc, cmi, cmg, ncc, nci, a50, a90, n],
                             device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            (loss_sum, top1_sum, top5_sum, top20_sum,
             anyhit_sum, anyhit20_sum, cov20_sum,
             cbc, cmc, cmi, cmg, ncc, nci, a50, a90, n) = t.tolist()
        n = max(int(n), 1)
        return {'loss': loss_sum / n, 'cov20': cov20_sum / n,
                'top1_acc': top1_sum / n,
                'top5_acc': top5_sum / n, 'top20_acc': top20_sum / n,
                'anyhit_acc': anyhit_sum / n,
                'anyhit20_acc': anyhit20_sum / n,
                'conf_best_correct': cbc / n, 'conf_mean_correct': cmc / n,
                'conf_max_incorrect': cmi / n, 'conf_margin': cmg / n,
                'n_conf_correct': ncc / n, 'n_conf_incorrect': nci / n,
                'conf_any50': a50 / n, 'conf_any90': a90 / n}

    def selection_score(val_m, dots_m):
        """The --select_on metric, always as lower-is-better.

        Drives best_model.pt AND (for --lr_schedule plateau) the LR cuts, so a
        run anneals on the same quantity it is judged by. conf_margin /
        conf_max_incorrect matter here because on this corpus val_loss goes
        NOISY-FLAT long before the separation metrics stop improving (measured:
        anyhit flat 0.9597-0.9612 from E24 while conf_margin kept rising to a
        run-best +0.5160 at E30), so selecting or annealing on val_loss stops
        responding to real progress.
        """
        so = args.select_on
        if so == 'val_loss':
            return val_m['loss']
        if so == 'val_top20':
            return -val_m.get('top20_acc', 0.0)
        if so == 'val_top1':
            return -val_m['top1_acc']
        if so == 'val_anyhit20':
            # any correct action inside the top-K=20 the beam expands
            return -val_m.get('anyhit20_acc', 0.0)
        if so == 'val_anyhit':
            # "top-1 is A reducing action", not agreement with the one
            # arbitrarily-recorded pick
            return -val_m.get('anyhit_acc', 0.0)
        if so == 'conf_margin':
            # best-correct score minus best-incorrect score. Higher is better.
            return -val_m.get('conf_margin', 0.0)
        if so == 'conf_max_incorrect':
            # confidence on the best WRONG action. Lower is better already.
            return val_m.get('conf_max_incorrect', 1.0)
        if so in ('extra_top20', 'dots_top20'):
            return -(dots_m or val_m).get('top20_acc', 0.0)
        if so in ('extra_top1', 'dots_top1'):
            return -(dots_m or val_m)['top1_acc']
        raise ValueError(f'unhandled --select_on {so!r}')

    log("\nStarting training...")
    log("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        if hasattr(train_dataset, 'set_epoch'):
            train_dataset.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_raw = train_epoch(
            model, train_loader, optimizer, device, epoch,
            log_every=args.log_every if is_main else 0,
            total_batches=approx_train_batches,
            max_iters=train_iter_cap,
        )
        val_raw = evaluate(model, val_loader, device, max_iters=val_iter_cap)
        dots_raw = (evaluate(model, dots_loader, device)
                    if dots_loader is not None else None)
        # NOTE: scheduler.step() moved below the aggregation --
        # ReduceLROnPlateau needs the val_loss it monitors, and that value must
        # be the DDP-aggregated one so every rank steps identically (stepping
        # on per-rank losses would drift the LR apart across ranks).

        # Sample-weighted aggregation across DDP ranks.
        train_m = aggregate_metrics(train_raw)
        val_m = aggregate_metrics(val_raw)
        dots_m = aggregate_metrics(dots_raw) if dots_raw is not None else None

        # ONE score drives BOTH checkpoint selection and the plateau scheduler,
        # so they can never disagree about what "better" means. Always
        # lower-is-better (accuracy-style metrics are negated).
        _score = selection_score(val_m, dots_m)
        if args.lr_schedule == 'plateau':
            scheduler.step(_score)
        else:
            scheduler.step()

        wall = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']
        log(f"Epoch {epoch}/{args.epochs} ({wall:.1f}s, lr={cur_lr:.2e}):")
        log(f"  Train: loss={train_m['loss']:.4f}, top1={train_m['top1_acc']:.4f}, top5={train_m['top5_acc']:.4f}")
        log(f"  Val:   loss={val_m['loss']:.4f}, top1={val_m['top1_acc']:.4f}, "
            f"top5={val_m['top5_acc']:.4f}, TOP20={val_m.get('top20_acc', 0):.4f}")
        # anyhit/cov20 score against the FULL valid set, so they stay meaningful
        # when the objective does not target the one recorded label. conf_* are
        # the only metrics that see CONFIDENCE rather than rank.
        log(f"  Any:   anyhit={val_m.get('anyhit_acc', 0):.4f}, "
            f"anyhit20={val_m.get('anyhit20_acc', 0):.4f}, "
            f"cov20={val_m.get('cov20', 0):.4f}")
        # ANY-correct confidence is the headline: the beam only needs ONE
        # correct action to be ranked and trusted, never all of them.
        log(f"  ANY:   conf_best_correct={val_m.get('conf_best_correct', 0):.4f}  "
            f"[>0.5 in {val_m.get('conf_any50', 0):.1%} of states, "
            f">0.9 in {val_m.get('conf_any90', 0):.1%}]")
        log(f"  Conf:  max_incorrect={val_m.get('conf_max_incorrect', 0):.4f}, "
            f"margin={val_m.get('conf_margin', 0):+.4f}, "
            f"mean_correct(ALL, fyi)={val_m.get('conf_mean_correct', 0):.4f}")
        log(f"         n>0.5: correct={val_m.get('n_conf_correct', 0):.2f}, "
            f"incorrect={val_m.get('n_conf_incorrect', 0):.2f}")
        if dots_m is not None:
            log(f"  Extra[{os.path.basename(args.extra_val)}]: "
                f"loss={dots_m['loss']:.4f}, top1={dots_m['top1_acc']:.4f}, "
                f"top5={dots_m['top5_acc']:.4f}, TOP20={dots_m.get('top20_acc', 0):.4f}")

        # Selection metric. best_val_loss holds the score being tracked; for
        # accuracy-style criteria we store the NEGATIVE so "lower is better"
        # stays uniform (and stays compatible with resumed checkpoints).
        is_best = _score < best_val_loss
        if is_best:
            best_val_loss = _score

        if is_main:
            # Always save last.pt — atomic via temp-then-rename.
            state = _full_state(epoch, val_m, train_m)
            tmp = output_dir / 'last.pt.tmp'
            torch.save(state, tmp)
            os.replace(tmp, output_dir / 'last.pt')

            if is_best:
                torch.save(state, output_dir / 'best_model.pt')
                # Name the metric actually used. The message used to say "val
                # loss" regardless of --select_on, which is misleading in the
                # logs of a run selecting on, say, val_anyhit.
                log(f"  -> New best {args.select_on}! Saved best_model.pt")

            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                torch.save(state, output_dir / f'checkpoint_epoch{epoch}.pt')

            if log_fp is not None:
                _d = (f"\t{dots_m['loss']:.6f}\t{dots_m['top1_acc']:.6f}\t{dots_m['top5_acc']:.6f}"
                      if dots_m is not None else "\t\t\t")
                _d += f"\t{val_m.get('top20_acc', 0):.6f}"
                _d += (f"\t{dots_m.get('top20_acc', 0):.6f}"
                       if dots_m is not None else "\t")
                # appended LAST so existing column positions are unchanged and
                # every earlier epochs.tsv stays parseable by the same scripts
                _d += f"\t{val_m.get('anyhit_acc', 0):.6f}"
                _d += f"\t{val_m.get('anyhit20_acc', 0):.6f}"
                _d += f"\t{val_m.get('cov20', 0):.6f}"
                _d += f"\t{val_m.get('conf_best_correct', 0):.6f}"
                _d += f"\t{val_m.get('conf_mean_correct', 0):.6f}"
                _d += f"\t{val_m.get('conf_max_incorrect', 0):.6f}"
                _d += f"\t{val_m.get('conf_margin', 0):.6f}"
                _d += f"\t{val_m.get('n_conf_correct', 0):.6f}"
                _d += f"\t{val_m.get('n_conf_incorrect', 0):.6f}"
                _d += f"\t{val_m.get('conf_any50', 0):.6f}"
                _d += f"\t{val_m.get('conf_any90', 0):.6f}"
                log_fp.write(f"{epoch}\t{train_m['loss']:.6f}\t{train_m['top1_acc']:.6f}\t{train_m['top5_acc']:.6f}"
                             f"\t{val_m['loss']:.6f}\t{val_m['top1_acc']:.6f}\t{val_m['top5_acc']:.6f}"
                             f"{_d}\t{cur_lr:.6e}\t{wall:.1f}\n")
                log_fp.flush()

        if ddp_enabled:
            # Make sure non-main ranks wait for rank-0 to finish writing before
            # the next epoch starts (purely defensive — no shared file system race
            # is possible here, but the barrier keeps the per-epoch logs ordered).
            dist.barrier()

    if is_main:
        torch.save(_full_state(args.epochs, val_m, train_m), output_dir / 'final_model.pt')
        if log_fp is not None:
            log_fp.close()
        log(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
