#!/usr/bin/env python
"""Topology selection for the reduction/routing layer.

SAILIR_TOPOLOGY environment variable keys everything; DEFAULT = pentagonbox so
all existing pentagonbox behavior is bit-identical with the flag unset. Set it
identically for the orchestrator AND all workers of a run (it rides the condor
environment= line next to SAILIR_SECTOR_RANK).

Each entry:
  N_IND / N_DEN        index slots / denominator slots (sector-mask bits)
  TOPO_DIR             Topology.from_dir input
  CANON_PKL            canonical sectors {rep_of, canonical} (sector-senior rank)
  CANON_MAPS_PKL       composite non-canonical -> canonical corner maps cache
  CANONICALIZE_MOD     module providing _transforms/image_unsigned/_reduce/P
                       (pentagonbox: momentum-engine 'canonicalize';
                        gravity3L:  store-backed 'canonicalize_GR')
"""
import os

# ROOT resolution (2026-09-06). The literal below is the canonical checkout on
# the production cluster and stays the default THERE, so nothing about that
# machine's behavior changes. It is not a valid path on every box the code now
# runs on (the p101 training/DAgger host is /home/shih/work/SAILIR_p101), and a
# stale ROOT is silent: TOPO_DIR/CANON_PKL/STORE_PKL just point at files that do
# not exist, and the failure surfaces far away as a FileNotFoundError inside
# canonicalize2's module-level pickle.load.
#
# Order: explicit SAILIR_ROOT > the production literal if it exists on this box
# > the repo this file lives in (<repo>/reduction/topo_config.py -> <repo>).
_LEGACY_ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get('SAILIR_ROOT') or (
    _LEGACY_ROOT if os.path.isdir(_LEGACY_ROOT) else _REPO_ROOT)
TOPOLOGY = os.environ.get('SAILIR_TOPOLOGY')
if TOPOLOGY is None:
    # 2026-08-03: the silent pentagonbox default produced wrong-topology
    # measurements twice (equations solving to None, diagnostics comparing
    # None==None -> fabricated conclusions). Defaulting now requires explicit
    # opt-in; unset is a hard error so wrong-family runs die at import.
    if os.environ.get('SAILIR_ALLOW_DEFAULT_TOPOLOGY', '0') == '1':
        TOPOLOGY = 'pentagonbox'
    else:
        raise RuntimeError(
            "SAILIR_TOPOLOGY is not set. Set SAILIR_TOPOLOGY=<family> "
            "(e.g. gravity3L / pentagonbox), or explicitly opt into the "
            "legacy default with SAILIR_ALLOW_DEFAULT_TOPOLOGY=1.")

_CFG = {
    'pentagonbox': dict(
        N_IND=11, N_DEN=8,
        TOPO_DIR=os.path.join(ROOT, "topology_input/pentagonbox"),
        CANON_PKL=os.path.join(ROOT, "results/canonical_sectors_tkey.pkl"),
        CANON_MAPS_PKL=os.path.join(ROOT, "results/sector_canon_maps.pkl"),
        # legacy momentum-engine provider stays PRODUCTION for pentagonbox
        # until the general engine's pentagonbox gate is signed off
        # (STORE_PKL is the general-engine store used by the gate scripts).
        CANONICALIZE_MOD='canonicalize',
        STORE_PKL=os.path.join(ROOT, "results/pentagonbox_transforms_v2.pkl"),
    ),
    'gravity3L': dict(
        N_IND=15, N_DEN=10,
        TOPO_DIR=os.path.join(ROOT, "topology_input/gravity3L"),
        # v2 = clean-den convention (298 canonical sectors; the ing-based v1
        # pkl over-merged 6 sectors via flip maps — see canonicalize_GR.py)
        CANON_PKL=os.path.join(ROOT, "results/canonical_sectors_GR_v2.pkl"),
        # PRODUCTION = the GENERAL engine (canonicalize2 + symmetry_engine2
        # store), behavioral-gated 2026-07-18 (run_engine2_behavioral_gate.sh:
        # orbits 1023/1023, canon maps 725/725, masters 45/23/1, router smoke
        # 27/27, FIRE-oracle 0 false zeros). The GR-specific modules
        # (symmetry_engine_GR / canonicalize_GR / sector_canon_maps_GR.pkl)
        # are retained as the gated reference implementation only.
        CANON_MAPS_PKL=os.path.join(ROOT, "results/sector_canon_maps_GR_engine2.pkl"),
        CANONICALIZE_MOD='canonicalize2',
        STORE_PKL=os.path.join(ROOT, "results/gravity3L_transforms_v2.pkl"),
    ),
}
if TOPOLOGY not in _CFG:
    raise RuntimeError(f"SAILIR_TOPOLOGY={TOPOLOGY!r} unknown (have {sorted(_CFG)})")

_c = _CFG[TOPOLOGY]
N_IND = _c['N_IND']
N_DEN = _c['N_DEN']
TOPO_DIR = _c['TOPO_DIR']
CANON_PKL = _c['CANON_PKL']
STORE_PKL = _c['STORE_PKL']
# A/B-only overrides (behavioral gating of the general engine): swap the
# canonicalize provider and redirect the composite-map cache so gate runs
# never clobber production pkls. NOT for production runs.
CANONICALIZE_MOD = os.environ.get('SAILIR_CANONICALIZE', _c['CANONICALIZE_MOD'])
CANON_MAPS_PKL = os.environ.get('SAILIR_CANON_MAPS_PKL', _c['CANON_MAPS_PKL'])


def canonicalize_module():
    """Import and return the topology's canonicalize provider module."""
    import importlib
    return importlib.import_module(CANONICALIZE_MOD)
