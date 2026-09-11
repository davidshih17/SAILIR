import pickle, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
TRIV=set(int(x) for x in open(BASE+"/results/kira_reduce_161/sectormappings/TA/trivialsector").read().split(",") if x.strip())
def secnum(i):
    n=0
    for k in range(8):
        if i[k]>0: n|=(1<<k)
    return n
for t in ["gate_small","m1_6prop","m2_4prop_dots","m3_5prop_deg3"]:
    for m in ["baseline","design1"]:
        p=f"{BASE}/results/ab_symmetry/{t}/{m}/reduction.pkl"
        if not os.path.exists(p): print(f"  {t:14s} {m:8s} : (no pkl / killed)"); continue
        d=pickle.load(open(p,"rb"))
        fe={k:v for k,v in d['final_expr'].items() if v!=0}
        nm=[k for k in fe if secnum(k) not in () ]  # count non-master-ish: all remaining terms
        # a term is "unreduced" if it's not a master; approximate by counting all final terms > masters
        scaleless=sum(1 for k in fe if secnum(k) in TRIV)
        print(f"  {t:14s} {m:8s} : jobs={d['total_jobs']:5d}  final_terms={len(fe):4d}  scaleless={scaleless:4d}  routed={d.get('n_symmetry_routed',0)}")
