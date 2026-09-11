import sys, os
sys.path.insert(0, "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2")
sys.path.insert(0, "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/reduction")
import pickle
p = sys.argv[1]
with open(p,'rb') as f: d = pickle.load(f)
bs = d['best_state']
print("start_integral:", d.get('start_integral'))
print("target_sector:", d.get('target_sector'), " start_w12:", d.get('start_w12'))
print("best_state.expr:", dict(bs.expr) if hasattr(bs,'expr') else bs)
print("best_state.n_non_masters:", getattr(bs,'n_non_masters',None))
print(f"\nbest_state.path ({len(bs.path)} steps):")
for i, step in enumerate(bs.path):
    tgt, op, delta = step
    print(f"  step {i+1}: target={tgt} op={op} delta={delta}")
