import sys, os, pickle
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE)
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
def load(tag):
    d=pickle.load(open(f"results/corner_reductions/cph_{tag}/reduction.pkl","rb"))
    return {tuple(k):v for k,v in (d.get('final_expr') or d.get('reduction') or d.get('masters')).items()}
A=load("A"); B=load("B")
print(f"I[2,1,1,0,1,1,...] -> {len(A)} masters: {sorted(A.items())[:4]}")
print(f"I[1,1,2,0,1,1,...] -> {len(B)} masters: {sorted(B.items())[:4]}")
print(f"\nARE THEY EQUAL (placeholder relation holds)? {A==B}")
if A!=B:
    diff={k:(A.get(k,0)-B.get(k,0))%1009 for k in set(A)|set(B)}; diff={k:v for k,v in diff.items() if v%1009}
    print(f"  differ by {len(diff)} terms: {sorted(diff.items())[:4]}")
