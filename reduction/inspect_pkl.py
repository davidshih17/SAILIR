#!/usr/bin/env python
import sys, pickle
p=sys.argv[1]
with open(p,'rb') as f: d=pickle.load(f)
print("type:", type(d))
if isinstance(d, dict):
    for k,v in d.items():
        t=type(v).__name__
        n = len(v) if hasattr(v,'__len__') else ''
        print(f"  {k!r}: {t}  len={n}")
        # peek at a couple entries for list/dict of integrals
        if k in ('resolved_subs','master_coeffs','final_expr','masters') and hasattr(v,'__iter__'):
            it=iter(v.items() if isinstance(v,dict) else enumerate(v))
            for i,(kk,vv) in zip(range(2), it):
                print(f"       sample key={kk!r}  -> {str(vv)[:80]}")
