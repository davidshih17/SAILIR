#!/usr/bin/env python
"""Print SAILIR's exact numeric kinematic point for pentagon-box, so the symmetry
engine evaluates its coefficients at the identical mod-p values as the IBPs."""
import sys, os
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
from sailir.topology import Topology

topo = Topology.from_dir(os.path.join(ROOT, "topology_input/pentagonbox"))
print("kinematic_invariants:", topo.kinematic_invariants)
print("kinematics_values   :", topo.kinematics_values)
print("n_indices           :", topo.n_indices)
print("n_denominators      :", topo.n_denominators)
print("n_actions (IBP+LI)  :", topo.n_actions)
