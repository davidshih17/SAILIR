import sys, os, pickle, glob
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE)
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
for pth in sorted(glob.glob("results/corner_reductions/c*/reduction.pkl")):
    d=pickle.load(open(pth,'rb'))
    tgt=tuple(d.get('start_integral'))
    expr=d.get('final_expr') or d.get('reduction') or d.get('masters')
    expr={tuple(k):v for k,v in expr.items()}
    selfmaster = (len(expr)==1 and tgt in expr)
    print(f"  {tgt} -> {len(expr)} term(s){'  [MASTER]' if selfmaster else '  [reduces: '+str(list(expr.items())[:2])+']'}")
