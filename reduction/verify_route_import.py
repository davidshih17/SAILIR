import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from symmetry_route import symmetry_rule
# with filter removed, target now routes to box + scaleless survivor (worker kills it later)
r=symmetry_rule((1,-1,1,0,1,1,0,0,0,0,0))
print("symmetry_rule(target) =", r)
print("has is_zero attr:", hasattr(sys.modules['symmetry_route'],'is_zero'))
