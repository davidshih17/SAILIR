import pickle, sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from symmetry_route import symmetry_rule
OUT=BASE+"/reduction/ibp_cmp"
ints=[(0,-1,2,0,2,-1,0,1,-1,0,0),(0,-1,1,0,2,0,-1,2,0,0,-1),(1,-1,1,0,0,0,0,3,-1,0,0)]
print(f"{'integral':40s} {'IBP-step terms':>15} {'symmetry terms':>15}")
for i,I in enumerate(ints,1):
    d=pickle.load(open(f"{OUT}/ibp_{i}.pkl","rb"))
    fe=d.get('final_expr',d.get('expr')); nib=len({k:v for k,v in fe.items() if v!=0}) if fe else 0
    r=symmetry_rule(I); nsym=len(r) if r is not None else -1
    print(f"{str(list(I)):40s} {nib:>15} {(nsym if nsym>=0 else 'survivor'):>15}")
