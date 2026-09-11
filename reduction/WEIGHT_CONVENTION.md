# The weight() convention — what it is, and the bug it used to carry

**`weight(i)` is `(r, s, tuple(-|a_i|))`. All three components run the SAME way:
LARGER = HIGHER = eliminated first. `max(expr, key=weight)` is the highest
integral.**

```python
r = sum of positive indices          # denominator powers ("dots")
s = sum of |negative indices|        # numerator powers
third = tuple(-abs(x) for x in i)    # NEGATED, so it points the same way as r, s
```

`weight(i)[0]` is `r` and `weight(i)[1]` is `s`, unchanged from before.

## The bug this replaced

`weight` used to return `(r, s, +|abs|)`. The first two components meant
larger = higher, but the third meant **smaller = higher** — a relative sign
mismatch *inside a single key*. No single comparison direction was correct, so
every ordering site had to hand-write

```python
(-weight(x)[0], -weight(x)[1], weight(x)[2])     # negate two, not the third
```

and any site that forgot was silently wrong. Nothing crashed; the numbers just
came out in the wrong order. Concretely, on 2026-09-02 the SAME unscramble
corpus measured **45% / 55% rise-fall with the raw tuple and 98% / 2% with the
correct order** — the difference was entirely the |abs| tiebreak reordering
integrals inside a fixed (r, s) shell.

The mismatch had also been rediscovered independently at least twice: one call
site hand-wrote `[-a for a in weight(x)[2]]`, and `build_recovery_data_v3._wkey`
carried a docstring explaining that "it cannot be a plain max(): the third
element is a tuple, so it can neither be negated nor compared in the same
direction as the first two". Both are now plain `max(..., key=weight)`.

## Rules

* **Never** re-sign weight's components. If you are writing `(-w[0], -w[1], ...)`
  you are rebuilding a key by hand; use `max`/`min` on `weight` directly, or
  `reverse=True`.
* **All** weight-like definitions in the tree must agree. There are several
  (`sailir.ibp_env.weight`, `generate_multisector_data.weight`, `replay.weight`,
  `reorder_filter_list_TA.weight`, and local `tw()` in five `reduction/` files).
  `results/truth/closures/reaudit_3_runtime.py` compares them BY VALUE and must
  report all agreeing.
* **`tkey` is deliberately different** and must NOT be derived from weight.
  `tkey` is `(-sector_rank, -r, -s, +|abs|)` with SMALLER = higher. Its `+|abs|`
  is correct *for tkey*. `_target_key` in the beam files computes it directly
  from the indices; deriving it from `weight` would need a per-component sign
  fix, which is the mismatch this change removed.
* **`gm` means `generate_multisector_data`, never `ibp_env`.** It used to mean
  both depending on the file, so `gm.weight` silently referred to different
  function objects. The `ibp_env as gm` alias is gone; do not reintroduce it.

## Verification harness

Any change touching weight or an ordering key must pass all of these:

| tool | what it proves |
|---|---|
| `results/truth/closures/weight_golden.py` | 11 locked decisions on a fixed 600-integral pool: tkey order, reference order, max_w12, rs pairs, strip decision, data-gen target pick. Diff BEFORE vs AFTER. |
| `results/truth/closures/ab_endtoend.py` | rebuilds real closures on Condor and compares field-wise against ones built by the previous code — the full pipeline, not a synthetic pool |
| `reaudit_1_ast.py` | AST walk: definition shapes, mixed-sign keys, weight in comparisons |
| `reaudit_2_text.py` | textual: direction misuse (`min` vs `max`), positive |abs| in a larger=higher key |
| `reaudit_3_runtime.py` | executes: all definitions agree by value; weight is a valid total order; ordering identical to pre-refactor |

The golden caught a live bug introduced during this very change: the total-order
strip in `ibp_env.get_raw_equation` compared a positive `|abs|` tuple against the
now-negated threshold and would have stripped nearly every boundary term, in the
raw-equation path every beam and closure build uses. It would not have thrown.
