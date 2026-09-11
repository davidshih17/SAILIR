import pickle
from collections import Counter
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
d=pickle.load(open(BASE+"/results/ab_symmetry/m2_4prop_dots/design1/reduction.pkl","rb"))
sizes=Counter(len(v) for v in d['cache'].values())
print("terms-per-cache-rule distribution:")
for sz,c in sorted(sizes.items())[:15]:
    print(f"    {sz} terms: {c} rules")
big=max(d['cache'].items(), key=lambda kv: len(kv[1]))
print(f"\nlargest rule: I{list(big[0])} -> {len(big[1])} terms")
