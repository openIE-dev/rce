"""RCEdb graph index (spec E.8.1; G-1..G-6 proposed in E.11). matplotlib, text as paths."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
ink = "#0f172a"
fig, ax = plt.subplots(figsize=(12, 5.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")
def box(x, y, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
def arrow(a, b, c="#334155", lw=1.4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=c, lw=lw))
ax.text(0.2, 4.9, "D: hash-chained log (the base record, a linear graph)", fontsize=13, weight="bold", color=ink)
ev = [("e1 Put(a)", "hash h1"), ("e2 Put(b)", "prev h1"), ("e3 Link(a,b)", "prev h2"), ("e4 Delete(b)", "prev h3")]
for i, (t, s) in enumerate(ev):
    x = 0.2 + i * 1.9
    box(x, 3.9, 1.5, 0.7, "#eef3fb", "#3366cc")
    ax.text(x + 0.75, 4.33, t, ha="center", fontsize=10, color=ink); ax.text(x + 0.75, 4.05, s, ha="center", fontsize=8.5, color="#475569")
    arrow((x + 1.5, 4.25), (x + 1.9, 4.25))
ax.text(7.9, 4.2, "append only (HS-1 to HS-5)", fontsize=10, color=ink)
arrow((4.0, 3.85), (4.0, 3.2))
ax.text(4.12, 3.45, "projection: a fold of the step over a log prefix, written in RCE (HS-7)", fontsize=10, color=ink)
ax.text(0.2, 2.85, "Primary index: typed property graph, as of e3 (G-1, proposed)", fontsize=13, weight="bold", color=ink)
for cx, t, s in [(1.2, "a: Item", "by e1"), (3.8, "b: Item", "by e2")]:
    ax.add_patch(Circle((cx, 1.85), 0.5, fc="#f0fbf0", ec="#2f9e44", lw=1.4))
    ax.text(cx, 1.9, t, ha="center", fontsize=10, color=ink); ax.text(cx, 1.65, s, ha="center", fontsize=8, color="#475569")
arrow((1.7, 1.85), (3.3, 1.85), "#2f9e44", 2)
ax.text(2.5, 2.0, "Link (made by e3)", ha="center", fontsize=9, color=ink)
ax.text(0.3, 0.75, "every node and edge carries the id of the event that made it (G-2)", fontsize=9.5, color=ink)
ax.text(0.3, 0.45, "tables and documents are further projections of the same log (G-6)", fontsize=9.5, color=ink)
arrow((4.4, 1.85), (6.75, 1.85))
ax.text(5.6, 2.05, "pattern: typed RCE value (G-3)", ha="center", fontsize=9, color=ink)
box(6.8, 0.45, 5.0, 2.05, "#fbf6ee", "#cc8833")
ax.text(7.0, 2.15, "Queries", fontsize=13, weight="bold", color=ink)
for i, t in enumerate(["Match(Node(Item) -[Link]-> Node(Item), bound 4)", "result: Set[Row] or Bag[Row], canonical order (G-4)",
                       "total: bounded paths or stratified fixpoint (G-5)", "as of e2: the same query over a shorter prefix (HS-9)"]):
    ax.text(7.0, 1.78 - i * 0.36, t, fontsize=9.5, color=ink)
fig.savefig("rcedb-graph.svg", bbox_inches="tight", facecolor="white"); fig.savefig("rcedb-graph.png", dpi=150, bbox_inches="tight", facecolor="white")
