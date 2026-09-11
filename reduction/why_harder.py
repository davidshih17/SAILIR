#!/usr/bin/env python
"""Test the anti-correlation: among the integrals the BASELINE reduced, do the ones
symmetry CAN reduce (reducible) cost less IBP time than the ones it can't (survivors)?
If yes: symmetry removes the cheap/redundant work and leaves the hard cores for workers."""
import pickle, glob, sys, os, statistics as st
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"; sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env; from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox"))); ibp_env.set_prime(1009)
from symmetry_route import symmetry_rule

# baseline worker time + steps per integral
bt={}
for p in glob.glob(f"{BASE}/results/ab_symmetry/m2_4prop_dots/baseline/work/results/*.pkl"):
    try:
        w=pickle.load(open(p,"rb"))
        if isinstance(w,dict) and 'time' in w:
            bt[tuple(w['original_integral'])]=(w['time'], w.get('steps',0))
    except Exception: pass

red_t=[]; red_s=[]; sur_t=[]; sur_s=[]
for I,(t,s) in bt.items():
    r=symmetry_rule(I)
    if r is not None: red_t.append(t); red_s.append(s)
    else:             sur_t.append(t); sur_s.append(s)

def summ(name,ts,ss):
    if not ts: print(f"  {name}: none"); return
    print(f"  {name:28s} n={len(ts):4d} | baseline time med={st.median(ts):5.2f}s mean={st.mean(ts):6.2f}s "
          f"sum={sum(ts):7.0f}s | steps med={st.median(ss):3.0f} mean={st.mean(ss):5.1f}")
print(f"m2: of {len(bt)} baseline-reduced integrals, classified by symmetry:")
summ("SYMMETRY-REDUCIBLE (routed)", red_t, red_s)
summ("SURVIVOR (must go to worker)", sur_t, sur_s)
if red_t and sur_t:
    print(f"\n  => survivors cost {st.mean(sur_t)/st.mean(red_t):.1f}x the mean IBP time of reducible ones,")
    print(f"     and reducible integrals are {100*sum(red_t)/sum(bt_t for bt_t,_ in bt.values()):.0f}% of baseline CPU but "
          f"{100*len(red_t)/len(bt):.0f}% of the count.")
