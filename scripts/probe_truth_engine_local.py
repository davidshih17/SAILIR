#!/usr/bin/env python
"""Can TruthEngine be constructed on THIS box?

topo_config.ROOT is hardcoded to the other server, so _tc.TOPO_DIR and the
CANON_*/STORE_* pkl paths under it do not exist here. The beam never noticed
because it takes --topology from the CLI and reads only N_DEN from topo_config.

This probe answers three separate questions:
  1. does the import chain (symmetry_route -> canonicalize2 -> store pkl) load?
  2. does TruthEngine construct against the LOCAL topology dir?
  3. does worth_replay actually run on a real target?
"""
import os, sys, traceback

os.environ.setdefault('SAILIR_TOPOLOGY', 'gravity3L')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['SAILIR_KEEP_SYSTEMS'] = '1'

ROOT = '/home/shih/work/SAILIR_p101'
sys.path.insert(0, os.path.join(ROOT, 'reduction'))
sys.path.insert(0, ROOT)

import topo_config as _tc
print('[topo_config]')
print('  ROOT        =', _tc.ROOT, '  exists:', os.path.isdir(_tc.ROOT))
print('  TOPO_DIR    =', _tc.TOPO_DIR, '  exists:', os.path.isdir(_tc.TOPO_DIR))
for k in ('CANON_PKL', 'CANON_MAPS_PKL', 'STORE_PKL'):
    v = getattr(_tc, k)
    print(f'  {k:14s}=', v, '  exists:', os.path.exists(v))
print('  CANONICALIZE_MOD =', _tc.CANONICALIZE_MOD)

LOCAL_TOPO = os.path.join(ROOT, 'topology_input/gravity3L')
print('\n[local topology dir]', LOCAL_TOPO, 'exists:', os.path.isdir(LOCAL_TOPO))
TRIV = os.path.join(LOCAL_TOPO, 'kira_validate/sectormappings/GR/trivialsector')
print('[trivialsector]', TRIV, 'exists:', os.path.exists(TRIV))

print('\n[1] importing truth_engine ...')
try:
    from truth_engine import TruthEngine
    print('    OK')
except Exception:
    traceback.print_exc()
    sys.exit(1)

print('\n[2] constructing TruthEngine on the LOCAL topology dir ...')
try:
    eng = TruthEngine(LOCAL_TOPO, TRIV if os.path.exists(TRIV) else None)
    print('    OK  n_actions =', eng.n_actions, ' trivial sectors =', len(eng.trivial))
except Exception:
    traceback.print_exc()
    sys.exit(2)

print('\n[3] worker_replay on the verify target ...')
T = (1, 0, 2, 0, 1, 2, 2, 0, 1, 1, 0, 0, 0, 0, -1)
try:
    wr = eng.worker_replay(T, verbose=False)
    print('    OK  keys =', sorted(wr))
    acts = wr.get('actions')
    print('    n_actions in replay =', len(acts) if acts is not None else None)
    if acts:
        print('    first row =', acts[0])
except Exception:
    traceback.print_exc()
    sys.exit(3)

print('\nALL THREE STAGES PASSED -- on-demand closure regeneration is viable locally.')
