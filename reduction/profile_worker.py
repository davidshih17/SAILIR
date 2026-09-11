#!/usr/bin/env python
"""What fraction of a worker's step time can the eval_coeff optimization reach?

Runs greedy_worker's real search under cProfile with the live campaign's flags
and a step cap, then reports cumulative time in get_raw_equation / eval_coeff
against the total. This settles by measurement whether the 2.60x on
get_raw_equation is a campaign-wide win or a builder-local one -- the action
scorer already bypasses eval_coeff via _tc_lin_table, but action ENUMERATION
still builds raw equations on cache miss, so reading the code is not enough.
"""
import cProfile
import io
import os
import pstats
import sys

R = '/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2'
sys.path.insert(0, R)
sys.path.insert(0, R + '/reduction')

TARGET = os.environ.get('PROF_INTEGRAL', '0,-10,-6,1,1,1,1,1,0,1,-1,-1,0,0,0')
STEPS = os.environ.get('PROF_STEPS', '120')

sys.argv = [
    'greedy_worker.py',
    '--topology', R + '/topology_input/gravity3L',
    '--integral=' + TARGET,
    '--output', '/tmp/claude-706/-het-p4-dshih/'
                '422e8a7a-68ad-4e51-a91e-bd72ad2e933b/scratchpad/prof_worker.pkl',
    '--model-checkpoint',
    R + '/checkpoints/gravity3L_p101_scratch/best_model.pt',
    '--max_steps', STEPS,
    '--prime', '101',
    '--device', 'cpu',
    '-v',
    '--v7-cpus', '1',
    '--no-paper-masters-only',
]

import greedy_worker

pr = cProfile.Profile()
pr.enable()
try:
    greedy_worker.main()
except SystemExit:
    pass
finally:
    pr.disable()

st = pstats.Stats(pr)
total = st.total_tt
print(f"\n=== PROFILE: {total:.1f}s total ===", flush=True)

want = ('get_raw_equation', 'eval_coeff', 'enumerate_valid_actions',
        'substitute', 'model_fwd', 'forward')
seen = {}
for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
    for w in want:
        if w in name:
            k = seen.setdefault(w, [0, 0.0, 0.0])
            k[0] += nc
            k[1] += tt
            k[2] = max(k[2], ct)
print(f"  {'function':<28} {'ncalls':>10} {'tottime':>9} {'cumtime':>9}  {'%tot':>6}")
for w in want:
    if w in seen:
        nc, tt, ct = seen[w]
        print(f"  {w:<28} {nc:>10,} {tt:>8.2f}s {ct:>8.2f}s  {100.0*ct/total:>5.1f}%")

s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(18)
print("\n=== top by tottime ===")
print('\n'.join(s.getvalue().splitlines()[4:30]))
