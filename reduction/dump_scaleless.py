import pickle, os
OUT="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/reduction/scaleless_test"
labels=["SPUR I[0,0,0,0,1,1,0,1] paper","SPUR nopaper",
        "A I[0,0,1,0,1,1] paper","A nopaper",
        "D I[1,0,0,0,1,1] paper","D nopaper"]
for i,lab in enumerate(labels,1):
    d=pickle.load(open(f"{OUT}/scal_{i}.pkl","rb"))
    fe=d.get('final_expr', d.get('expr'))
    fe={k:v for k,v in fe.items() if v!=0} if fe else fe
    print(f"  {lab:38s} -> final_expr = {fe if fe else '{} (ZERO)'}")
