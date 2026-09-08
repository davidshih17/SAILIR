# greedy_reduce — the hardened greedy width-1 reducer

`reduction/greedy_reduce.py` is `reduction/beam_search_v9.py` with everything
the greedy configuration cannot reach removed: **5,676 -> 2,428 lines**, in 15
phases, each validated **125/125 bit-identical** against the reference run
(path, success, final_expr, steps) with peak RSS unchanged.

## The algorithm

Per step, on ONE state:

    target   = min(non_masters, key=_target_key)
    valid    = _tc_upenum_valid(target, state)     # enumerate legal actions
    valid    = _v9_cull(valid, ..., max_actions=1000)
    row      = model probabilities over valid
    children = [apply_action(state, target, *valid[i]) for i in argsort(-row)[:20]]
    if any child is a success: take it
    state    = min(children, key=lambda c: (-(c.score - 0.1*c.n_non_masters),
                                            c.max_w12, c.n_non_masters))

It is **not a beam search**. Nothing survives a step boundary except one state.
The 20 exist only within a step, because the selection key needs `nm`, which is
observable only AFTER applying an action. That is the price of the NM penalty,
not a search frontier.

## Interface

5 required args, 16 keyword args (none dead), 9 environment variables.
`top_k=20` and `max_actions=1000` are stated in the signature, not in the
environment.

## Guards

The build refuses to start outside its certified configuration. Eight semantic
flags are pinned to the values OBSERVED in the certified runs, and **unset
fails too** — an unset `SAILIR_SECTOR_RANK` falls back to `'0'`, a different
ordering. `max_actions != 1000` aborts: the cull cap decides which candidates
the model ever sees.

These are assertions, not folded constants. A constant I had merely inferred
would pass every test today and be wrong later; every phase was safe precisely
because a mistake changed results and got caught.

## Reproducing any build

    scripts/strip_greedy.py scripts/strip_specs/p8.json --hits <trace json>

Specs hold ORIGINAL `beam_search_v9.py` line numbers, so they compose and every
build is regenerable from the untouched original. That file is committed here;
`git show <this commit>:reduction/beam_search_v9.py` is the exact input.

## Gates (scripts/)

| check | catches |
|---|---|
| `strip_greedy.py` | parse failure; `expect` pins the original text of every replaced line |
| `check_dangling.py` | names read but no longer assigned — counts REPLACED lines, not just deleted |
| `check_fences.py` | removal of load-bearing blocks (the stall `break`, the cull fallback, resume) |
| `check_optimizations.py` | a lost optimization — calibrated against the reference, not hand-picked floors |
| `check_worker_refs.py` | cross-file: module attributes the worker reads |
| one-target smoke test | everything static analysis cannot — it found the worker banner reading a deleted attribute, and `top_k=None` overriding a new default |

## What was deliberately KEPT

The trace called all of these "dead" because the 125 targets happened to
succeed. Reading them showed otherwise:

- `if not tasks:` — owns the `break` that exits a stalled search
- `if scored is None:` — the compiled cull's documented fallback (3 live `return None` sites)
- resume + checkpoint — the orchestrator's straggler path; checkpoints only
  looked dead because both traced targets ran <100 steps and the interval is 100
- `add_sub_to_resolved` dict path — core algebra, packed-gated

## Backward compatibility

`beam_search_v9.py` is untouched — the pipeline only ever reads it. The
orchestrator, `beam_search_truthcull.py` and `onestep_worker_v9.py` keep every
feature (tabu, iraws, beam sorts, fork pool). The greedy chain is a parallel
stack: its own module, worker, and wrapper.

## Timing note

Ratios from a DENSE 125-job submit are unreliable: Condor packs the batch onto
few machines (once 61 of 125 onto one 64-core box), and jobs throttle each
other. Observed ratios up to 3.14x that returned to 0.98x when re-run one job
per machine. Bit-identity is immune to this; only timing needs care.
