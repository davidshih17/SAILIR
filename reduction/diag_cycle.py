import pickle, sys
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE)
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(BASE+"/topology_input/pentagonbox")); ibp_env.set_prime(1009)
from sailir.ibp_env import set_paper_masters_only, is_master
set_paper_masters_only(True)
d=pickle.load(open(BASE+"/results/ab_symmetry/m2_4prop_dots/design1/reduction.pkl","rb"))
fe={k:v for k,v in d['final_expr'].items() if v!=0}
cache=d['cache']
nm=[k for k in fe if not is_master(k)]
print(f"final_expr: {len(fe)} terms, {len(nm)} non-masters")
in_cache=[k for k in nm if k in cache]
print(f"non-masters still in final_expr that ARE cache keys: {len(in_cache)} / {len(nm)}")
# check for 2-cycles: I->{...J...}, J->{...I...}
cyc=0; examples=[]
for I in in_cache:
    for J in cache[I]:
        if J in cache and I in cache[J]:
            cyc+=1
            if len(examples)<3: examples.append((I,J))
            break
print(f"non-masters in a 2-cycle (I->..J.. and J->..I..): {cyc}")
for I,J in examples:
    print(f"  CYCLE: I{list(I)} -> {dict(list(cache[I].items())[:3])}...")
    print(f"         J{list(J)} -> {dict(list(cache[J].items())[:3])}...")
    # are they same total-lex-weight?
    def tw(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0), tuple(-abs(x) for x in i))  # LARGER = higher on all 3 components (matches ibp_env.weight)
    print(f"         tw(I)={tw(I)[:2]} tw(J)={tw(J)[:2]}  I<J:{tw(I)<tw(J)} J<I:{tw(J)<tw(I)}")
