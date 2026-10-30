"""Proposed (E.11, not normative): tensor unification of primitives. matplotlib, text as paths."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
fig, ax = plt.subplots(figsize=(14, 8)); ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
ink="#1f2937"
def box(x,y,w,h,t,fc,fs=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.12",fc=fc,ec="#475569",lw=1.3))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=ink)
def arr(a,b,t=None):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=15,lw=1.3,color="#475569"))
    if t: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+0.12,t,ha="center",fontsize=9,color="#334155")
ax.text(0.3,7.55,"Tensor unification of primitives (PROPOSED, spec E.11, not normative)",fontsize=16,weight="bold",color=ink)
ax.text(0.3,7.15,"Every primitive value is Tensor[elem, shape]; shape is part of the type and checked before the program runs.",fontsize=10,color="#475569")
# rank ladder
ranks=[("rank 0: scalar\nTensor[f64, []]",1),("rank 1: vector\nTensor[i64, [n]]",4),("rank 2: matrix\nTensor[f64, [m, n]]",12),("rank 3+\nTensor[u8, [h, w, 3]]",27)]
for i,(t,k) in enumerate(ranks):
    x=0.4+i*2.0
    import math
    if i==0: ax.add_patch(Rectangle((x+0.7,5.6),0.35,0.35,fc="#93c5fd",ec="#1e3a8a"))
    elif i==1:
        for j in range(4): ax.add_patch(Rectangle((x+0.2+j*0.35,5.6),0.35,0.35,fc="#93c5fd",ec="#1e3a8a"))
    elif i==2:
        for r in range(3):
            for j in range(4): ax.add_patch(Rectangle((x+0.2+j*0.35,5.25+r*0.35),0.35,0.35,fc="#93c5fd",ec="#1e3a8a"))
    else:
        for d in (0.24,0.12,0):
            for r in range(3):
                for j in range(3): ax.add_patch(Rectangle((x+0.25+j*0.35+d,5.25+r*0.35+d),0.35,0.35,fc="#bfdbfe",ec="#1e3a8a",lw=0.7))
    ax.text(x+0.9,4.75,t,ha="center",va="center",fontsize=9.5,color=ink)
ax.text(4.3,6.75,"One canonical value form for numbers and homogeneous data",ha="center",fontsize=11,weight="bold")
box(8.6,5.0,5.0,1.7,"Records over named axes (where homogeneous)\n{x: f64, y: f64, z: f64}  \u2192  Tensor[f64, [axis: {x, y, z}]]\nheterogeneous records stay records","#dcfce7",9.5)
# checker
box(0.4,2.4,4.0,1.5,"Checker: shapes in types\nadd: [m, n] x [m, n] \u2192 [m, n]\nmatmul: [m, k] x [k, n] \u2192 [m, n]\n[m, k] x [j, n], k \u2260 j: requires change","#fee2e2",9.5)
box(5.0,2.4,4.0,1.5,"Reductions: fixed order (N-9)\nsum over axis = left fold unless the\nprogram names a fixed tree; thread count\nand fabric never change the bits","#fef3c7",9.5)
box(9.6,2.4,4.0,1.5,"Placement (HC-12): cost table keyed by\n(pure op, shape, elem)\nCPU, GPU, FPGA chosen by cost;\nresult bits identical on every fabric","#e0f2fe",9.5)
arr((2.4,4.4),(2.4,3.9)); arr((4.4,3.15),(5.0,3.15)); arr((9.0,3.15),(9.6,3.15))
box(0.4,0.4,13.2,1.3,"Stay separate canonical values (ragged, not tensors): Text, Map, graphs, sum types (enums, Option, Result).\nA list of texts or a map of lists is a value of those types, not a tensor; no padding or implicit densification.","#f1f5f9",10)
fig.savefig("tensor-unification.svg",bbox_inches="tight",facecolor="white"); fig.savefig("tensor-unification.png",dpi=150,bbox_inches="tight",facecolor="white")
