#!/usr/bin/env python
"""Precise weight accounting of routing rules.

For each sampled rule I -> {J: c}:
  rs(I)   = (r, s) = (sum of positive exps, sum of |negative| exps)
  compare max over RHS of (r, s) vs source; count any RAISES (should be none),
  how many rules have exactly-equal top (r,s), and the share of RHS terms
  strictly below the source in (r,s).
Also: for equal-(r,s) top terms, is the |a| exponent multiset preserved?
"""
import pickle
import sys

ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/reduction")

from sailir import ibp_env
from sailir.topology import Topology
import topo_config as _tc
ibp_env.init_from_topology(Topology.from_dir(_tc.TOPO_DIR))


def rs(I):
    return (sum(x for x in I if x > 0), sum(-x for x in I if x < 0))


with open(ROOT + "/results/gr_reduce/g1023/work/sym_memo.pkl", "rb") as f:
    tab = pickle.load(f)
rules = [(k, v) for k, v in tab.items() if v]

step = max(1, len(rules) // 2000)
raises = top_equal = top_lower = 0
terms_below = terms_equal = 0
multiset_same = multiset_diff = 0
for k, v in rules[::step][:2000]:
    k_rs = rs(k)
    rhs_rs = [rs(t) for t in v]
    mx = max(rhs_rs)
    if mx > k_rs:
        raises += 1
    elif mx == k_rs:
        top_equal += 1
    else:
        top_lower += 1
    terms_equal += sum(1 for x in rhs_rs if x == k_rs)
    terms_below += sum(1 for x in rhs_rs if x < k_rs)
    if mx == k_rs:
        # exponent multiset of one equal-top term vs source
        tops = [t for t, x in zip(v, rhs_rs) if x == k_rs]
        if any(sorted(t) == sorted(k) for t in tops):
            multiset_same += 1
        else:
            multiset_diff += 1

n = raises + top_equal + top_lower
tt = terms_below + terms_equal
print(f"sample {n} rules:", flush=True)
print(f"  RHS max (r,s) RAISED above source: {raises}", flush=True)
print(f"  RHS max (r,s) EQUAL to source:     {top_equal}", flush=True)
print(f"  RHS max (r,s) STRICTLY LOWER:      {top_lower}", flush=True)
print(f"  RHS terms at source (r,s): {terms_equal}, "
      f"strictly below: {terms_below} "
      f"({100*terms_below/tt:.0f}% of {tt})", flush=True)
print(f"  equal-top rules with exponent multiset preserved: "
      f"{multiset_same}/{multiset_same+multiset_diff}", flush=True)
