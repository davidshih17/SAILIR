#!/usr/bin/env python
"""Merge banked one-step worker results from finished arms into a new arm's
work/results dir (hard links; dedup by integral, first source wins). One-step
reductions are basis- and target-independent, so any arm can reuse them.

Usage: merge_arm_caches.py DEST_TAG SRC_TAG [SRC_TAG...]
"""
import os, sys, glob

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/gr_reduce"
dest_tag, src_tags = sys.argv[1], sys.argv[2:]
dest = os.path.join(ROOT, dest_tag, "work/results")
os.makedirs(dest, exist_ok=True)

have = set()
for f in glob.glob(dest + "/*.pkl"):
    have.add("_".join(os.path.basename(f).split("_")[2:]))

n_new = n_dup = 0
for tag in src_tags:
    for f in sorted(glob.glob(os.path.join(ROOT, tag, "work/results/*.pkl"))):
        key = "_".join(os.path.basename(f).split("_")[2:])   # integral part
        if key in have:
            n_dup += 1
            continue
        have.add(key)
        os.link(f, os.path.join(dest, os.path.basename(f)))
        n_new += 1
print(f"{dest_tag}: linked {n_new} results from {src_tags} "
      f"({n_dup} duplicates skipped); total {len(have)}")
