"""Test file format of rce-conformance (rce-test/1).

A test is a .rce file. Its metadata is a comment block at the END of the
file, so a test can still control its first line (`# rce-version:`, a BOM)
and every byte before the block:

    #--- rce-test
    # id: lex/tab-indent
    # clauses: [W-LEX-1]
    # area: lexing
    # expect: fit               ok | fit | run | trace
    # diagnostic: W-LEX-1       fit: the rule code of the first diagnostic
    # events: Inc, Dec          run/trace: the inputs (else fn main runs)
    # fuel: 40                  run: the fuel limit
    # schedule_seeds: [1, 2]    run: also run under these par schedules; all must agree
    # outcome: trap:...         run: done | kept:E | stopped:E | trap:KIND | deadlock
    # outputs:
    #   - emit 3
    # trace_head: <sha256>      trace: hash of the last log entry
    # steps: [{...}]            host: batches, sourced offers and window closes (JSON)
    # host_expect: {...}        host: state_hash, entries (host.* kinds that depend on
    #                           the machine are dropped), admitted (JSON)
    # flags: [generated]
    # provenance: harvested spec/examples/01_counter.rce
    #---

The source given to an implementation is the file's bytes up to the line
`#--- rce-test`, with that line and everything after it removed.
"""
import json, re

MARK = "#--- rce-test"

def split(raw: str):
    i = raw.rfind("\n" + MARK)
    if i < 0:
        if raw.startswith(MARK):
            i = -1
        else:
            raise ValueError("no `#--- rce-test` block")
    src = raw[: i + 1] if i >= 0 else ""
    meta_txt = raw[i + 1 :] if i >= 0 else raw
    return src, parse_meta(meta_txt)

def scalar(v):
    v = v.strip()
    if v.startswith("{") or v.startswith("[{") or v.startswith("[\""):
        return json.loads(v)
    if v.startswith("[") and v.endswith("]"):
        return [scalar(x) for x in v[1:-1].split(",") if x.strip()]
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if len(v) >= 2 and v[0] == v[-1] == '"':
        return json.loads(v)
    return v

def parse_meta(txt):
    meta, key = {}, None
    for l in txt.splitlines()[1:]:
        if l.strip() == "#---":
            break
        body = l[1:] if l.startswith("#") else l
        if body.startswith("   - ") or body.startswith("  - "):
            item = body.split("- ", 1)[1]
            meta.setdefault(key, []).append(json.loads(item) if item.startswith('"') else item)
            continue
        m = re.match(r"\s*([a-z_]+):\s*(.*)$", body)
        if m:
            key = m.group(1)
            meta[key] = scalar(m.group(2)) if m.group(2).strip() else []
    return meta

def render_meta(meta):
    order = ["id", "area", "clauses", "expect", "diagnostic", "events", "fuel", "schedule_seeds",
             "outcome", "outputs", "trace_head", "steps", "tear", "host_expect", "nodes", "script", "replicate_expect", "flags", "provenance", "note"]
    out = [MARK]
    for k in order + sorted(set(meta) - set(order)):
        if k not in meta or meta[k] is None:
            continue
        v = meta[k]
        if k == "outputs":
            out.append("# outputs:")
            for x in v:
                s = x if (x == x.strip() and x and not x.startswith('"')) else json.dumps(x)
                out.append(f"#   - {s}")
        elif k in ("steps", "host_expect", "script", "replicate_expect"):
            out.append(f"# {k}: {json.dumps(v, separators=(',', ':'))}")
        elif isinstance(v, list):
            out.append(f"# {k}: [{', '.join(str(x) for x in v)}]")
        else:
            out.append(f"# {k}: {v}")
    out.append("#---")
    return "\n".join(out) + "\n"

def write(path, src, meta):
    if src and not src.endswith("\n"):
        src += "\n"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(src + render_meta(meta))
