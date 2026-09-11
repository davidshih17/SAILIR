#!/usr/bin/env python
"""Can the {0,3,4,5} 2D block {num@1,num@9} be reduced to a single LEX-TOTAL-WEIGHT
representative? Order them by total-weight (w1,w2,|abs|); a canonicalization to one
representative needs the symmetry action to be TRIANGULAR in that order (higher ->
higher + strictly lower). If some symmetry maps the LOWER-total-weight member up to a
combination containing the HIGHER one, no single-representative reduction exists."""
import sys, os
BASE="/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0,BASE); sys.path.insert(0,os.path.join(BASE,"reduction"))
from sailir import ibp_env
from sailir.topology import Topology
ibp_env.init_from_topology(Topology.from_dir(os.path.join(BASE,"topology_input/pentagonbox_nosym"))); ibp_env.set_prime(1009)
from canonicalize import _transforms, image_unsigned
P=1009; ND=8
def sec(i): return tuple(1 if i[k]>0 else 0 for k in range(ND))
def totalweight(i): return (sum(x for x in i if x>0), sum(-x for x in i if x<0), tuple(-abs(x) for x in i))  # LARGER = higher on all 3 components (matches ibp_env.weight)
combo=(0,3,4,5); corner=tuple(1 if k in combo else 0 for k in range(11)); csec=sec(corner)
u1=tuple(-1 if k==1 else corner[k] for k in range(11))   # num@1
u9=tuple(-1 if k==9 else corner[k] for k in range(11))   # num@9

# total-weight order: HIGHER |abs| (lex) = LOWER total-weight (sub-weight, per v8 strip)
# so compare (w1,w2) ascending, then |abs| DESCENDING gives ascending total-weight.
def tw_key(i):
    w=totalweight(i); return (w[0],w[1],tuple(-x for x in w[2]))   # ascending total-weight
lo,hi = (u1,u9) if tw_key(u1)<tw_key(u9) else (u9,u1)
print(f"block members (both (w1,w2)=(4,1)):")
print(f"  num@1 = {u1}  |abs|={totalweight(u1)[2]}")
print(f"  num@9 = {u9}  |abs|={totalweight(u9)[2]}")
print(f"  total-weight order: LOWER = {'num@1' if lo==u1 else 'num@9'}, HIGHER = {'num@9' if hi==u9 else 'num@1'}")
print(f"  => canonicalization would reduce the HIGHER to c*(LOWER) + strictly-lower\n")

autos=[t for t in _transforms(corner) if (lambda im: im and len(im)==1 and sec(next(iter(im)))==csec)(image_unsigned(corner,*t))]
# same-weight 2x2 action of each g on {lo, hi}; check triangularity in total order (lo first)
def samewt_coeff(img, target):
    return img.get(target,0)%P if img else 0
nontri=[]
for idx,t in enumerate(autos):
    imlo=image_unsigned(lo,*t); imhi=image_unsigned(hi,*t)
    if imlo is None or imhi is None: continue
    # projection onto {lo,hi}
    a=samewt_coeff(imlo,lo); b=samewt_coeff(imlo,hi)   # g(lo)= a*lo + b*hi + (other/lower)
    c=samewt_coeff(imhi,lo); d=samewt_coeff(imhi,hi)   # g(hi)= c*lo + d*hi + (other)
    # triangular (in total order, LOWER=lo is the reduction target/last): to reduce HIGHER->LOWER,
    # need g to NOT send lo (lower) up into hi (higher).  b!=0 means lo -> ... + hi  == UP-mixing.
    if b%P!=0:
        nontri.append((idx,a,b,c,d))
print(f"checked {len(autos)} symmetries. Symmetries that map the LOWER member UP into the HIGHER (b!=0):")
if not nontri:
    print("  NONE -> action is triangular in total order -> CAN reduce to a single representative")
else:
    for idx,a,b,c,d in nontri[:6]:
        print(f"  sym#{idx}: g(LOWER) = {a}*LOWER + {b}*HIGHER + (lower);  g(HIGHER)= {c}*LOWER + {d}*HIGHER + (lower)")
    print(f"\n  {len(nontri)} symmetries push the lower-total-weight member UP into the higher one.")
    print("  => NO single-representative reduction in total-weight order: the two are irreducibly mixed.")
    print("     (Consistent with the verified 2D absolute irreducibility: no order triangularizes it.)")
