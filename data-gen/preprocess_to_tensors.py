#!/usr/bin/env python3
"""
Preprocess JSONL training data to packed tensor files.

Uses packed format with offsets for variable-length data.
Small dtypes to minimize file size.

KEEPS ALL FIELDS FROM THE ORIGINAL JSONL:
- scramble_id, sector_id, sector_mask, step
- target, target_weight
- expr, subs, valid_actions
- num_valid_actions, chosen_action, chosen_action_idx
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

# Make sailir.topology importable when this script runs directly.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from sailir.topology import Topology


def load_and_convert(input_paths, n_indices, max_samples=None):
    """Load JSONL and convert to lists of tensors. KEEPS ALL FIELDS.

    input_paths can be a single str/Path or a list of paths — they are
    processed in order without concatenation, which avoids a giant intermediate
    file when sharding a large dataset.

    n_indices is the integral-tuple length (from the topology); used for
    shape fallbacks when subs / actions are empty.
    """
    samples = []

    if isinstance(input_paths, (str, Path)):
        input_paths = [input_paths]

    i_global = 0
    for input_path in input_paths:
        if max_samples and i_global >= max_samples:
            break
        # EXP 1: the START target T of this trajectory. The model is given the
        # per-step target (the current top of the bucket) but never T, yet the
        # whole task is defined relative to T -- "active work" means at-or-above
        # T in the total order, so T is the floor of the active region and two
        # states with identical expressions but different T have different
        # amounts of work left. T needs no re-recording: at step 0 the
        # expression is {T: 1}, so the first line's target IS T.
        # Scramble/replay files have no start target (they walk up from a
        # master with no fixed T); there the per-line fallback below sets
        # start_target = the line's own target, which keeps those samples in
        # distribution rather than feeding an out-of-range sentinel.
        start_target = None
        _file_start = len(samples)      # EXP 2: where this trajectory begins
        with open(input_path) as f:
            for line in f:
                if max_samples and i_global >= max_samples:
                    break
                s = json.loads(line)
                expr = s['expr']
                subs = s['subs']
                actions = s['valid_actions']
                if start_target is None:
                    start_target = s.get('start_target', s['target'])

                samples.append({
                    # === METADATA ===
                    'scramble_id': s.get('scramble_id', -1),
                    'sector_id': s.get('sector_id', -1),
                    'sector_mask': torch.tensor(s['sector_mask'], dtype=torch.int8),
                    'step': s.get('step', -1),

                    # === TARGET INFO ===
                    'target': torch.tensor(s['target'], dtype=torch.int8),
                    'target_weight': torch.tensor(s.get('target_weight', [0, 0])[:2], dtype=torch.int8),
                    'start_target': torch.tensor(start_target, dtype=torch.int8),

                    # === EXPRESSION ===
                    'expr_integrals': torch.tensor([t[0] for t in expr], dtype=torch.int8),
                    'expr_coeffs': torch.tensor([t[1] for t in expr], dtype=torch.int16),

                    # === SUBSTITUTIONS (complex structure: [integral, [[integral,coeff],...]] for each) ===
                    'sub_integrals': torch.tensor([t[0] for t in subs], dtype=torch.int8) if subs else torch.zeros(0, n_indices, dtype=torch.int8),
                    'subs_raw': subs,

                    # === ACTIONS ===
                    'action_ibp_ops': torch.tensor([a[0] for a in actions], dtype=torch.int8),
                    'action_deltas': torch.tensor([a[1] for a in actions], dtype=torch.int8),
                    'num_valid_actions': s.get('num_valid_actions', len(actions)),

                    # === LABEL ===
                    'chosen_action_ibp_op': s['chosen_action'][0] if 'chosen_action' in s else (actions[s['chosen_action_idx']][0] if actions else 0),
                    'chosen_action_delta': torch.tensor(s['chosen_action'][1] if 'chosen_action' in s else (actions[s['chosen_action_idx']][1] if actions else [0] * n_indices), dtype=torch.int8),
                    'label': s['chosen_action_idx'],

                    # EVERY unused closure row that solves this target, as
                    # indices into valid_actions. All of them are correct
                    # actions, so 'label' above is one arbitrary pick from this
                    # set. Carried through so (a) a different label rule can be
                    # applied offline without re-recording and (b) a model can
                    # be scored on "did top-1 hit ANY reducing action" instead
                    # of conformity to one convention. Empty list for the older
                    # dots_sweep_* recordings, which predate the field. Cheap:
                    # median 10 ints/step against ~9700 valid actions.
                    'valid_label_idxs': s.get('valid_label_idxs', []),
                })

                i_global += 1
                if i_global % 100000 == 0:
                    print(f"  Loaded {i_global} samples (file {Path(input_path).name})...", flush=True)

        # EXP 2: steps remaining until this trajectory closes, INCLUDING the
        # current step, so the final step is 1 (never 0 -> log1p stays sane).
        # Only known once the whole file is read, hence the back-fill. This is
        # the cost-to-go label the beam search has no signal for; it needs no
        # re-recording, it is just the trajectory length minus the step index.
        _n = len(samples) - _file_start
        for _j in range(_file_start, len(samples)):
            samples[_j]['steps_remaining'] = _n - (_j - _file_start)

    return samples


def pack_samples(samples, include_subs=True):
    """Pack variable-length samples into flat arrays with offsets.

    include_subs=False DROPS the substitution fields (sub_integrals,
    sub_offsets, subs_raw) entirely. They exist only for the original
    `IBPActionClassifier`, whose FullSubstitutionEncoder consumed them; the
    `nosubs` variant -- now the only model in use -- opens its forward() with
      del sub_keys, sub_repl_ints, sub_repl_coeffs, sub_repl_mask, sub_mask
    so every one of those tensors is built, read off disk, unpickled, collated,
    pinned, shipped to the GPU, and thrown away untouched.

    Measured on a real shard: subs_raw alone is 195 MB of 448 MB (43.6%), and
    all sub_* fields together are 44.6% of the packed bytes. That was a
    one-time startup cost under --data_dir, but the sharded loader re-reads and
    re-unpickles every shard EVERY EPOCH, so it became a recurring tax on the
    slowest part of the loop.

    Corpora packed with include_subs=False are usable ONLY by nosubs models.
    Recordings are kept, so a full-model corpus can always be repacked.
    """

    # Collect all data
    all_scramble_ids = []
    all_sector_ids = []
    all_sector_masks = []
    all_steps = []

    all_targets = []
    all_target_weights = []
    all_start_targets = []          # EXP 1
    all_steps_remaining = []        # EXP 2

    all_expr_integrals = []
    all_expr_coeffs = []
    expr_offsets = [0]

    all_sub_integrals = []
    sub_offsets = [0]
    all_subs_raw = []

    all_action_ibp_ops = []
    all_action_deltas = []
    action_offsets = [0]
    all_num_valid_actions = []

    all_chosen_action_ibp_ops = []
    all_chosen_action_deltas = []
    all_labels = []

    # MULTI-LABEL: every correct action per sample, flat + offsets (same layout
    # as expr/actions). Needed because 'labels' above is ONE arbitrary pick out
    # of ~18 equally correct actions, so hard cross-entropy penalises the model
    # for choosing any of the other ~17.
    all_valid_label_idxs = []
    valid_label_offsets = [0]

    for s in samples:
        # Metadata
        all_scramble_ids.append(s['scramble_id'])
        all_sector_ids.append(s['sector_id'])
        all_sector_masks.append(s['sector_mask'])
        all_steps.append(s['step'])

        # Target
        all_targets.append(s['target'])
        all_target_weights.append(s['target_weight'])
        # EXP 1: fall back to the per-step target for any sample packed
        # before start_target existed, so old caches stay loadable.
        all_start_targets.append(s.get('start_target', s['target']))
        # 0 marks "no trajectory length known" (e.g. replay/scramble samples
        # packed from elsewhere); the trainer MASKS those out of the value loss
        # rather than training on a fabricated target.
        all_steps_remaining.append(s.get('steps_remaining', 0))

        # Expression
        all_expr_integrals.append(s['expr_integrals'])
        all_expr_coeffs.append(s['expr_coeffs'])
        expr_offsets.append(expr_offsets[-1] + len(s['expr_integrals']))

        # Substitutions
        all_sub_integrals.append(s['sub_integrals'])
        sub_offsets.append(sub_offsets[-1] + len(s['sub_integrals']))
        all_subs_raw.append(s['subs_raw'])

        # Actions
        all_action_ibp_ops.append(s['action_ibp_ops'])
        all_action_deltas.append(s['action_deltas'])
        action_offsets.append(action_offsets[-1] + len(s['action_ibp_ops']))
        all_num_valid_actions.append(s['num_valid_actions'])

        # Label
        all_chosen_action_ibp_ops.append(s['chosen_action_ibp_op'])
        all_chosen_action_deltas.append(s['chosen_action_delta'])
        all_labels.append(s['label'])

        # Fall back to the single label when a sample predates the field (the
        # old dots_sweep_* recordings), so a mixed corpus still packs and the
        # set loss degrades to ordinary cross-entropy on those rows rather
        # than seeing an empty candidate set and producing -log(0).
        _vl = s.get('valid_label_idxs') or [s['label']]
        all_valid_label_idxs.extend(int(x) for x in _vl)
        valid_label_offsets.append(valid_label_offsets[-1] + len(_vl))

    return {
        # === METADATA ===
        'scramble_ids': torch.tensor(all_scramble_ids, dtype=torch.int32),
        'sector_ids': torch.tensor(all_sector_ids, dtype=torch.int16),
        'sector_masks': torch.stack(all_sector_masks),
        'steps': torch.tensor(all_steps, dtype=torch.int16),

        # === TARGET INFO ===
        'target_integrals': torch.stack(all_targets),
        'target_weights': torch.stack(all_target_weights),
        'start_target_integrals': torch.stack(all_start_targets),   # EXP 1
        'steps_remaining': torch.tensor(all_steps_remaining, dtype=torch.int32),  # EXP 2

        # === EXPRESSION ===
        'expr_integrals': torch.cat(all_expr_integrals) if all_expr_integrals else torch.zeros(0, 7, dtype=torch.int8),
        'expr_coeffs': torch.cat(all_expr_coeffs) if all_expr_coeffs else torch.zeros(0, dtype=torch.int16),
        'expr_offsets': torch.tensor(expr_offsets, dtype=torch.int32),

        # === SUBSTITUTIONS (omitted entirely when include_subs=False) ===
        **({
            'sub_integrals': torch.cat(all_sub_integrals) if any(len(s) > 0 for s in all_sub_integrals) else torch.zeros(0, 7, dtype=torch.int8),
            'sub_offsets': torch.tensor(sub_offsets, dtype=torch.int32),
            'subs_raw': all_subs_raw,  # Full substitution expressions (list of lists)
        } if include_subs else {}),

        # === ACTIONS ===
        'action_ibp_ops': torch.cat(all_action_ibp_ops) if all_action_ibp_ops else torch.zeros(0, dtype=torch.int8),
        # NOTE: action_deltas shape (n, n_indices) — fallback uses the trailing
        # dim from the first non-empty delta tensor we saw, or 0 if all empty.
        'action_deltas': torch.cat(all_action_deltas) if all_action_deltas else torch.zeros(0, 0, dtype=torch.int8),
        'action_offsets': torch.tensor(action_offsets, dtype=torch.int32),
        'num_valid_actions': torch.tensor(all_num_valid_actions, dtype=torch.int32),

        # === LABEL ===
        'chosen_action_ibp_ops': torch.tensor(all_chosen_action_ibp_ops, dtype=torch.int8),
        'chosen_action_deltas': torch.stack(all_chosen_action_deltas),
        'labels': torch.tensor(all_labels, dtype=torch.int32),

        # === MULTI-LABEL (all correct actions per sample) ===
        'valid_label_idxs': torch.tensor(all_valid_label_idxs, dtype=torch.int32),
        'valid_label_offsets': torch.tensor(valid_label_offsets, dtype=torch.int32),
    }


def preprocess(input_path, output_dir, n_indices, val_split=0.1, test_split=0.1, seed=42):
    """Preprocess JSONL to packed tensor files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading samples from {input_path}...", flush=True)
    samples = load_and_convert(input_path, n_indices=n_indices)
    n_total = len(samples)
    print(f"Total samples: {n_total}", flush=True)

    # Shuffle and split
    random.seed(seed)
    random.shuffle(samples)

    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)

    splits = {
        'test': samples[:n_test],
        'val': samples[n_test:n_test + n_val],
        'train': samples[n_test + n_val:],
    }

    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}", flush=True)

    # Pack and save each split
    print("Packing and saving splits...", flush=True)
    for name, split_samples in splits.items():
        packed = pack_samples(split_samples)
        path = output_dir / f'{name}.pt'
        torch.save(packed, path)
        size_mb = path.stat().st_size / 1e6
        print(f"  {name}: {len(split_samples)} samples, {size_mb:.1f} MB", flush=True)
        print(f"    Keys: {list(packed.keys())}", flush=True)

    print("Done!", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', type=str, required=True,
                        help='Path to topology_input/<family>/ directory '
                             '(used to read n_indices for shape fallbacks)')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='One or more JSONL input files. Multiple files are '
                             'streamed in sequence (no intermediate concat).')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    topology = Topology.from_dir(args.topology)

    print("=" * 70, flush=True)
    print("Preprocess JSONL to Packed Tensor Files (ALL FIELDS PRESERVED)", flush=True)
    print("=" * 70, flush=True)
    print(f"  topology: {topology.name} (n_indices={topology.n_indices})", flush=True)
    for k, v in vars(args).items():
        print(f"  {k}: {v}", flush=True)
    print(flush=True)

    preprocess(args.input, args.output_dir, n_indices=topology.n_indices,
               val_split=args.val_split, test_split=args.test_split, seed=args.seed)


if __name__ == '__main__':
    main()
