"""RCE infographics: overview, host contract, RCEdb, status. matplotlib only.
Run: python3 make_infographics.py  (writes SVG + PNG next to this file)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "path"})
C = dict(form="#dbeafe", chk="#dcfce7", log="#fef3c7", emit="#ede9fe", host="#fee2e2", fab="#e0f2fe", ink="#1f2937", edge="#475569")

def box(ax, x, y, w, h, text, fc, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=C["edge"], lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=C["ink"], weight="bold" if bold else "normal", wrap=True)

def arrow(ax, a, b, text=None, style="-|>", ls="-", off=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=16, lw=1.4, color=C["edge"], ls=ls))
    if text:
        ax.text((a[0]+b[0])/2 + off[0], (a[1]+b[1])/2 + off[1], text, ha="center", fontsize=9, color="#334155")

def canvas(w, h, title, sub):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.text(0.3, h - 0.45, title, fontsize=16, weight="bold", color=C["ink"])
    ax.text(0.3, h - 0.85, sub, fontsize=10, color="#475569")
    return fig, ax

def save(fig, name):
    for e in ("svg", "png"):
        fig.savefig(f"{name}.{e}", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# (a) overview
fig, ax = canvas(14, 7.6, "RCE at a glance", "One program, one meaning: a Form. The checker admits it, the log records every step, emitters project it, a host runs it.")
box(ax, 0.3, 4.3, 2.4, 1.3, "source.rce\n(text, integrated\ndevelopment environment,\ntree-sitter)", "#f1f5f9", fs=9)
box(ax, 3.3, 4.0, 3.2, 1.9, "rce checker\nparse, types, ownership,\ntermination, rules named\non refusal", C["chk"], bold=False)
box(ax, 7.1, 3.6, 3.2, 2.7, "Form\nP = (S, E, \u03b4, D, \u03bb)\n\nS states   E events\n\u03b4 step: (s, e) \u2192 (s', out)\nD durable log\n\u03bb outputs \u2192 effects", C["form"], fs=11)
arrow(ax, (2.7, 4.95), (3.3, 4.95)); arrow(ax, (6.5, 4.95), (7.1, 4.95), "accepted")
ax.text(4.9, 3.6, "refused: names the rule", ha="center", fontsize=9, color="#b91c1c")
targets = ["Rust", "Scala", "Haskell", "Julia", "wasm"]
for i, t in enumerate(targets):
    y = 6.0 - i * 0.62
    box(ax, 11.4, y, 2.2, 0.5, t, C["emit"], fs=10)
    arrow(ax, (10.3, 4.95), (11.4, y + 0.25))
ax.text(12.5, 6.65, "rce emit targets", ha="center", fontsize=10, weight="bold")
box(ax, 7.1, 1.6, 3.2, 1.3, "Log D\nappend-only, SHA-256 chained;\nstate = fold of D", C["log"], fs=10)
arrow(ax, (8.7, 3.6), (8.7, 2.9), "every step", off=(0.75, -0.05))
box(ax, 3.3, 1.6, 3.2, 1.3, "Host\none admitter, adapters,\nreplay with adapters off", C["host"], fs=10)
arrow(ax, (6.5, 2.25), (7.1, 2.25), style="<|-|>")
fabrics = ["CPU", "GPU", "FPGA", "browser", "embedded"]
for i, f in enumerate(fabrics):
    box(ax, 0.3 + i * 1.32, 0.25, 1.2, 0.6, f, C["fab"], fs=9)
ax.text(3.6, 1.02, "compute fabrics (bit-exact ops only; others enter as logged events)", ha="center", fontsize=9)
arrow(ax, (4.9, 1.6), (4.9, 1.2))
save(fig, "rce-overview")

# (b) host contract
fig, ax = canvas(14, 7.4, "The RCE host contract (spec Appendix E)", "Five primitives every surveyed target supplies; everything else is an adapter whose result is logged.")
prims = ["1  Pure compute on bytes\n(integer ops, binary64 RNE)", "2  Append-only storage", "3  SHA-256 as pure code", "4  One input queue,\none output queue", "5  Monotonic counter\n(read by the admitter only)"]
for i, p in enumerate(prims):
    box(ax, 0.3, 5.2 - i * 1.05, 3.4, 0.85, p, C["chk"], fs=10)
ax.text(2.0, 6.25, "Primitives", ha="center", weight="bold")
adapters = ["keyboard", "socket", "file", "clock", "GPU", "foreign lib"]
for i, a in enumerate(adapters):
    box(ax, 4.6, 5.4 - i * 0.82, 2.0, 0.6, a, C["fab"], fs=10)
ax.text(5.6, 6.25, "Adapters", ha="center", weight="bold")
box(ax, 7.5, 3.0, 2.4, 1.6, "Admitter\n(HC-1: exactly one)\norders inputs,\nappends to D", C["host"], fs=10)
for i in range(6):
    arrow(ax, (6.6, 5.7 - i * 0.82), (7.5, 3.8))
box(ax, 10.7, 4.3, 3.0, 1.3, "Program: \u03b4 and \u03bb\nsees only admitted events", C["form"], fs=10)
box(ax, 10.7, 2.0, 3.0, 1.3, "Log D (durable append,\nhash chained, verify on read)", C["log"], fs=10)
arrow(ax, (9.9, 4.1), (10.7, 4.9), "events"); arrow(ax, (9.9, 3.5), (10.7, 2.7), "append")
arrow(ax, (12.2, 4.3), (12.2, 3.3), "\u03bb requests \u2192 one result event (HC-3)", style="<|-|>", off=(0, 0))
box(ax, 7.5, 0.4, 6.2, 1.1, "Replay (HC-8, T37): read D, disconnect every adapter, re-derive\nevery state hash. No clock, file, network, device or environment is read.", "#f1f5f9", fs=9)
arrow(ax, (12.2, 2.0), (12.2, 1.5))
save(fig, "host-contract")

# (c) RCEdb
fig, ax = canvas(14, 5.6, "RCEdb: the log is the database", "One primary artefact (D). Indexes are RCE code: deterministic folds over a prefix of D. Queries read indexes, with a defined order.")
box(ax, 0.3, 1.6, 3.4, 2.2, "Log D\nappend-only, SHA-256 chained\ndurable barrier per OS\nverify on read, drop torn tail", C["log"], fs=10)
idx = ["by key (map fold)", "by time (Tick order)", "full text / custom"]
for i, t in enumerate(idx):
    box(ax, 4.9, 3.4 - i * 1.0, 3.6, 0.75, "index = fold(D prefix)\n" + t, C["form"], fs=9.5)
    arrow(ax, (3.7, 2.7), (4.9, 3.75 - i * 1.0))
ax.text(6.7, 4.35, "Indexes written in RCE (example 55)", ha="center", weight="bold", fontsize=10)
box(ax, 9.7, 1.9, 2.4, 1.6, "Queries\nordered results;\nsame D prefix,\nsame answer", C["chk"], fs=10)
for i in range(3):
    arrow(ax, (8.5, 3.75 - i * 1.0), (9.7, 2.7))
box(ax, 12.4, 1.9, 1.4, 1.6, "digest\nper seq\n(rebuild\n& check)", "#f1f5f9", fs=9)
arrow(ax, (12.1, 2.7), (12.4, 2.7))
ax.text(0.3, 0.6, "Snapshots are caches: (seq, state_hash, bytes); a mismatch with D at seq discards the snapshot. Source: rce-host-survey RCEDB.md R1 to R6; spec 6.1, D.5, D.6.", fontsize=9, color="#475569")
save(fig, "rcedb")

# (d) status
rows = [("Scala emit tests", 55, 55, "tests passed"), ("Haskell emit, checked by GHC", 451037, 451037, "cases agreeing"),
        ("Julia probe", 200000, 200000, "cases, 0 mismatches"), ("IDE test suite", 811, 811, "tests passed")]
fig, ax = plt.subplots(figsize=(11, 4.2))
for i, (n, ok, tot, unit) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, 1.0, color="#e5e7eb", height=0.55); ax.barh(y, ok / tot, color="#16a34a", height=0.55)
    ax.text(0.02, y, f"{ok:,} / {tot:,} {unit}" if "Julia" not in n else f"0 / {tot:,} mismatches", va="center", color="white", weight="bold", fontsize=11)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=11)
ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("RCE status: test and differential results (October 2026)", loc="left", fontsize=14, weight="bold")
fig.text(0.01, 0.01, "Bar = share passing / agreeing. Counts as reported by each run; see the repository for the commands.", fontsize=9, color="#475569")
fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, "status")
