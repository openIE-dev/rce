#!/usr/bin/env python3
"""Lookup-table controls of the corpus programs (critique 5: Goodhart, RJ-F2).

A program's stated obligations about its steps are its `example` lines (6.9):
each folds a list of events from the initial state and states the outcome of
the last step. A *lookup-table control* of a program P is a program that
matches P exactly on the listed example event sequences and on nothing else:
for every region R of P it adds one field `tblpos: u64 = 0`, the position of
R in a trie of the example event lists restricted to the events R has a step
for; every step of R keeps P's pattern, guard and body but applies only at a
trie node that has a child for its event (and moves `tblpos` to that child);
one more step per event of R moves along the trie without touching P's
fields (P selecting no step of R there), and a last unguarded step fails with
the new error `TblOff`, so every event off the table keeps the state. Where P
has exactly one region with exactly one field, the arrow form and the
example form `-> v` are rewritten to their block and `f == (v)` meanings
(8.4, 6.9), since the control has two fields (an outcome that already
names f is a condition and stays as written).

For every admitted program of the corpus population (scripts/corpus.py) with
example lines, the script writes the control, runs `rce check` on it, checks
that on the example event lists the control's trace equals P's (outcome and
state, with `tblpos` projected away) and probes off-table event lists (each
example list followed by another example's list) for a difference from P.
It also counts the program's other obligations (`check` lines, `requires`
and `ensures` contracts, refined types) to say which, if any, constrain
behaviour beyond the examples.

    scripts/table-control.py --rce PATH/TO/rce --out DIR

writes DIR/controls.jsonl, DIR/summary.json and the controls under
DIR/controls/. The run is a pure function of the checkout (no randomness).
"""
import argparse, importlib.util, json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("corpus", ROOT / "scripts" / "corpus.py")
corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(corpus)

POS = "tblpos"
OFF = "TblOff"


def split_top(s, sep=","):
    """Split on `sep` outside brackets and strings."""
    out, depth, cur, q = [], 0, "", None
    i = 0
    while i < len(s):
        ch = s[i]
        if q:
            cur += ch
            if ch == "\\" and i + 1 < len(s):
                cur += s[i + 1]
                i += 1
            elif ch == q:
                q = None
        elif ch in "\"'":
            q = ch
            cur += ch
        elif ch in "([{":
            depth += 1
            cur += ch
        elif ch in ")]}":
            depth -= 1
            cur += ch
        elif ch == sep and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
        i += 1
    if cur.strip():
        out.append(cur.strip())
    return out


EXAMPLE = re.compile(r"^(\s*)example (.*?) -> (.*)$")


def examples(src):
    """(line index, indent, [event texts], outcome text) of every example line."""
    out = []
    for i, l in enumerate(src.split("\n")):
        m = EXAMPLE.match(l)
        if m:
            out.append((i, m.group(1), split_top(m.group(2)), m.group(3)))
    return out


def lead(ev):
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", ev)
    return m.group(1) if m else None


def blocks(lines):
    """Top-level items as (start, end) line ranges (end exclusive), with the head line kind."""
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        if l and not l[0].isspace():
            j = i + 1
            while j < len(lines) and (lines[j].startswith(" ") or lines[j].strip() == ""):
                j += 1
            # trailing blank lines belong to no item
            k = j
            while k > i + 1 and lines[k - 1].strip() == "":
                k -= 1
            out.append((i, k))
            i = j
        else:
            i += 1
    return out


STEP = re.compile(r"^step ([A-Za-z_][A-Za-z0-9_]*) on (.+?)(?: if (.+?))?(?: -> (.+))?$")


def regions(lines):
    """{region: [field names]} in declaration order."""
    out = {}
    for s, e in blocks(lines):
        m = re.match(r"^state ([A-Za-z_][A-Za-z0-9_]*)\s*$", lines[s])
        if m:
            out[m.group(1)] = [re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", l).group(1) for l in lines[s + 1:e] if re.match(r"\s*[A-Za-z_][A-Za-z0-9_]*\s*:", l)]
    return out


def control(src):
    """The lookup-table control of `src`, or (None, reason)."""
    lines = src.split("\n")
    regs = regions(lines)
    if not regs:
        return None, "no state region"
    exs = examples(src)
    if not exs:
        return None, "no example lines"
    bl = blocks(lines)
    steps = []  # (start, end, region, pattern, guard, arrow)
    for s, e in bl:
        if lines[s].startswith("step "):
            m = STEP.match(lines[s].rstrip())
            if not m:
                return None, f"step head not parsed: {lines[s]}"
            steps.append((s, e, m.group(1), m.group(2).strip(), m.group(3), m.group(4)))
    if not steps:
        return None, "no step"
    fields = {}
    for l in lines:
        m = re.match(r"^(?:[a-z]+ )*event ([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*?)\))?(?:\s+priority\s+\d+)?\s*$", l)
        if m:
            fields[m.group(1)] = [x.split(":")[0].strip() for x in split_top(m.group(2) or "")]

    def wild(ev):
        fs = fields.get(ev, [])
        return f"{ev}(" + ", ".join(f"{f}: _" for f in fs) + ")" if fs else ev
    single = len(regs) == 1 and len(next(iter(regs.values()))) == 1
    handled = {}
    for _, _, r, pat, _, _ in steps:
        handled.setdefault(r, [])
        if lead(pat) not in handled[r]:
            handled[r].append(lead(pat))
    # one trie per region over the example events it has a step for
    tries = {}
    for r, evs in handled.items():
        nodes = {(): 0}
        edges = {}  # event -> [(node, child)]
        for _, _, seq, _ in exs:
            path = ()
            for ev in seq:
                name = lead(ev)
                if name not in evs:
                    continue
                parent = nodes[path]
                path = path + (name,)
                if path not in nodes:
                    nodes[path] = len(nodes)
                    edges.setdefault(name, []).append((parent, nodes[path]))
        tries[r] = edges
    out = list(lines)
    # replace from the bottom so line numbers stay valid
    plan = []  # (start, end, new lines)
    for s, e, r, pat, guard, arrow in steps:
        ev = lead(pat)
        es = tries[r].get(ev, [])
        on = " or ".join(f"{POS} == {p}" for p, _ in es) if es else f"{POS} == {POS} + 1"
        g = f"({guard}) and ({on})" if guard else on
        if arrow is not None:
            f = regs[r][0] if len(regs[r]) == 1 else None
            if f is None:
                return None, "arrow form on a region with more than one field"
            body = [f"    {f} = {arrow}"]
        else:
            body = lines[s + 1:e]
        move = []
        for k, (p, c) in enumerate(es):
            move += [f"    {'if' if k == 0 else 'elif'} {POS} == {p}", f"        {POS} = {c}"]
        plan.append((s, e, [f"step {r} on {pat} if {g}"] + move + body))
    # after the last step of each (region, event): the trie move with P's fields untouched, then TblOff
    last = {}
    for s, e, r, pat, _, _ in steps:
        last[(r, lead(pat))] = (s, e, pat)
    tails = {}
    for (r, ev), (s, e, pat) in last.items():
        es = tries[r].get(ev, [])
        add = []
        if es:
            on = " or ".join(f"{POS} == {p}" for p, _ in es)
            add.append(f"step {r} on {wild(ev)} if {on}")
            for k, (p, c) in enumerate(es):
                add += [f"    {'if' if k == 0 else 'elif'} {POS} == {p}", f"        {POS} = {c}"]
        add += [f"step {r} on {wild(ev)}", f"    fail {OFF}"]
        tails[e] = tails.get(e, []) + add
    edits = {s: (e, new) for s, e, new in plan}
    res = []
    i = 0
    while i < len(out):
        if i in edits:
            e, new = edits[i]
            res += new
            if e in tails:
                res += tails[e]
            i = e
            continue
        res.append(out[i])
        i += 1
    # the position field, the error, and the example forms of a single-field program
    final = []
    for idx, l in enumerate(res):
        final.append(l)
        m = re.match(r"^state ([A-Za-z_][A-Za-z0-9_]*)\s*$", l)
        if m and m.group(1) in handled:
            final.append(f"    {POS}: u64 = 0")
    text = "\n".join(final)
    text = text.rstrip("\n") + f"\n\nerror {OFF}\n"
    if single:
        f = next(iter(regs.values()))[0]
        fixed = []
        for l in text.split("\n"):
            m = EXAMPLE.match(l)
            if m and not m.group(3).startswith("kept:") and not re.search(rf"\b{f}\b", m.group(3)):
                l = f"{m.group(1)}example {m.group(2)} -> {f} == ({m.group(3)})"
            fixed.append(l)
        text = "\n".join(fixed)
    return text, {"regions": list(regs), "handled": handled, "positions": {r: sum(len(v) for v in t.values()) for r, t in tries.items()}, "single_field": single}


def strip_obligations(src):
    """The program text with its example and check lines removed (so it can be traced on streams)."""
    return "\n".join(l for l in src.split("\n") if not re.match(r"\s*(example|check)\b", l))


def trace(rce, path, streams, tmp):
    """[(outcome, projected state)] per stream, or None when the program cannot be traced."""
    sf = tmp / "streams.txt"
    sf.write_text("\n".join(streams) + "\n")
    r = rce.run("trace", "--json", str(path), "--streams", str(sf))
    if r.returncode != 0:
        return None
    try:
        j = json.loads(r.stdout)
        out = [[(s["outcome"], s["hashed"]["state"]) for s in t["steps"]] for t in j["traces"]]
    except (ValueError, KeyError):
        return None
    # `rce trace` does not link the library modules a program `use`s (rce check does):
    # such a trace stops at a trap naming the module, and says nothing about behaviour
    if any("unknown name" in json.dumps(o) for t in out for o, _ in t):
        return None
    return out


def project(tr, pos_regions):
    """Drop the position field (the first field) of each region that has one."""
    if tr is None:
        return None
    out = []
    for t in tr:
        out.append([(o, [reg[1:] if k in pos_regions else reg for k, reg in enumerate(st)]) for o, st in t])
    return out


def obligations(src):
    lines = src.split("\n")
    return {
        "examples": sum(1 for l in lines if re.match(r"\s*example ", l)),
        "checks": sum(1 for l in lines if re.match(r"\s*check ", l)),
        "requires": sum(1 for l in lines if re.match(r"\s*requires ", l)),
        "ensures": sum(1 for l in lines if re.match(r"\s*ensures ", l)),
        "refined_types": len(re.findall(r"\bwhere\b|\{\s*self\b", src)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rce", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "controls").mkdir(parents=True, exist_ok=True)
    tmp = out / ".tmp"
    tmp.mkdir(exist_ok=True)
    rce = corpus.Rce(a.rce)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True).stdout.strip()
    cands = [(p, None) for p, _, _ in corpus.population()] + [(p, s) for p, s, _ in corpus.conformance(True)]
    seen, rows = set(), []
    for p, src in cands:
        rel = p.relative_to(ROOT).as_posix()
        src = src if src is not None else p.read_text(encoding="utf-8")
        orig = tmp / "orig.rce"
        orig.write_text(src, encoding="utf-8")
        if rce.run("check", str(orig)).returncode != 0:
            continue
        nl = rce.run("fmt", "--nameless", str(orig)).stdout
        if not nl.strip() or corpus.sha(nl) in seen:
            continue
        seen.add(corpus.sha(nl))
        exs = examples(src)
        if not exs:
            continue
        row = {"path": rel, "commit": commit, "obligations": obligations(src)}
        text, info = control(src)
        if text is None:
            row.update(outcome="not_synthesised", why=info)
            rows.append(row)
            continue
        cpath = out / "controls" / (rel.replace("/", "__"))
        cpath.write_text(text, encoding="utf-8")
        row["control"] = cpath.relative_to(out).as_posix()
        row.update(info)
        c = rce.run("check", str(cpath))
        msg = (c.stdout + c.stderr).strip()
        rules = sorted(set(re.findall(r"\bW-[A-Z0-9]+-\d+\b|\bW-PLACE\b", msg)))
        row["control_check"] = "admitted" if c.returncode == 0 else "requires_change"
        row["control_rules"] = rules
        row["control_message"] = msg[:400] if c.returncode else ""
        # validity: on the example lists the control equals P (projected)
        streams = [", ".join(seq) for _, _, seq, _ in exs]
        so, sc = tmp / "o.rce", tmp / "c.rce"
        so.write_text(strip_obligations(src), encoding="utf-8")
        sc.write_text(strip_obligations(text), encoding="utf-8")
        regs = info["regions"]
        pos_idx = {k for k, r in enumerate(regs) if r in info["handled"]}
        to = trace(rce, so, streams, tmp)
        tc = project(trace(rce, sc, streams, tmp), pos_idx)
        row["traceable"] = to is not None
        row["equal_on_examples"] = to is not None and tc is not None and to == tc
        # off-table probes: example i followed by example j (i, j over all pairs)
        probes = [streams[i] + ", " + streams[j] for i in range(len(streams)) for j in range(len(streams))]
        po = trace(rce, so, probes, tmp)
        pc = project(trace(rce, sc, probes, tmp), pos_idx)
        if po is None or pc is None:
            row["probe_differs"] = None
        else:
            row["probes"] = len(probes)
            row["probe_differs"] = sum(1 for x, y in zip(po, pc) if x != y)
        if row["control_check"] == "requires_change" and rules != ["W-INT-1"]:
            row["outcome"] = "synthesis_requires_change"
        elif not row["traceable"]:
            row["outcome"] = "control_" + row["control_check"] + "_untraceable"
        elif not row["equal_on_examples"]:
            row["outcome"] = "synthesis_not_equal"
        elif row["control_check"] == "admitted":
            row["outcome"] = "control_admitted"
        else:
            row["outcome"] = "obligations_distinguish"
        rows.append(row)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    with open(out / "controls.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    valid = [r for r in rows if r["outcome"] in ("control_admitted", "obligations_distinguish")]
    summary = {
        "commit": commit, "programs_with_examples": len(rows),
        "outcomes": {o: sum(1 for r in rows if r["outcome"] == o) for o in sorted({r["outcome"] for r in rows})},
        "valid_controls": len(valid),
        "admitted_of_valid": sum(1 for r in valid if r["outcome"] == "control_admitted"),
        "distinguished_of_valid": sum(1 for r in valid if r["outcome"] == "obligations_distinguish"),
        "valid_with_probe_difference": sum(1 for r in valid if r.get("probe_differs")),
        "valid_without_probe_difference": [r["path"] for r in valid if not r.get("probe_differs")],
        "valid_with_other_obligations": {k: sum(1 for r in valid if r["obligations"][k]) for k in ("checks", "requires", "ensures", "refined_types")},
        "not_valid": [{"path": r["path"], "outcome": r["outcome"], "why": r.get("why") or r.get("control_message", "")[:200]} for r in rows if r not in valid],
        "rce_cpu_s": round(rce.cpu, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
