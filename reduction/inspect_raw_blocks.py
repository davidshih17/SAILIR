import os
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
TA = os.path.join(ROOT, "results/kira_reduce_161/sectormappings/TA")
blocks, cur = [], []
for ln in open(os.path.join(TA, "symmetries")):
    s = ln.rstrip("\n")
    if s.strip() == "":
        if cur: blocks.append(cur); cur = []
    else:
        cur.append(s)
if cur: blocks.append(cur)
print(f"total blocks: {len(blocks)}")
from collections import Counter
print("block line-counts:", Counter(len(b) for b in blocks))
for k in (0, 1, 30):
    b = blocks[k]
    print(f"\n=== block {k}: {len(b)} lines ===")
    for i, ln in enumerate(b):
        print(f"  [{i}] {ln}")
