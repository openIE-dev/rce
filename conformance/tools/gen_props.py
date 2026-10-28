#!/usr/bin/env python3
"""Generated tests whose expected result is fixed by a law, not by a run.

  lex      mutations of admitted programs into ones a lexical or syntax rule
           reports (tab, bare CR, BOM, bad dedent, out-of-range literal, bad
           \\u{}, `unsafe`, dropped bracket, unknown name,
           duplicate fn). The intended rule is the expected code.
  T7       consistent renaming of event and error names keeps every log hash
           (name independence): the expected trace head is the original's.
  T16      `par` programs with disjoint branch footprints: every schedule
           gives the source-order result (schedule_seeds), and the same
           footprints made to overlap are reported by W-PAR-1.
  HC-15    sourced inputs offered in shuffled arrival orders, with
           retransmissions: the same D order and state hash (HC-15, HC-16).
  fuel     for each fn main program, the least fuel that completes it, k:
           at k it is done and at k-1 it traps (6.7 / T2: fuel decides only
           whether a run completes, never its value).
Wherever the reference disagrees with the law, the test keeps the law and
the case goes to findings/props.json.
  gen_props.py TESTS_DIR SEED
"""
import json, os, pathlib, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
sys.path.insert(0, os.path.dirname(__file__))
import rcetest
from refclient import Ref

out, seed = pathlib.Path(sys.argv[1]), int(sys.argv[2])
rnd = random.Random(seed); ref = Ref(); findings = []; count = {}

def put(area, sub, name, src, meta):
    d = out / area / sub; d.mkdir(parents=True, exist_ok=True)
    meta = {"id": f"{area}/{sub}/{name}", "area": area, **meta, "flags": ["generated"]}
    rcetest.write(d / f"{name}.rce", src, meta); count[sub] = count.get(sub, 0) + 1

accepted = []
for p in sorted(out.rglob("*-accept.rce")):
    src, meta = rcetest.split(p.read_text(encoding="utf-8"))
    if not meta.get("fixtures"):
        accepted.append((p.stem, src))

# ---- lex / syntax mutations
def mut_tab(s):
    ls = s.split("\n"); idx = [i for i, l in enumerate(ls) if l.startswith("    ")]
    if not idx: return None
    i = rnd.choice(idx); ls[i] = "\t" + ls[i][4:]; return "\n".join(ls)
def mut_cr(s):
    i = s.find("\n", len(s) // 3);  return None if i < 0 else s[:i] + "\r" + s[i:] if False else s[:i] + "\r " + s[i:]
def mut_bom(s): return "\ufeff" + s
def mut_dedent(s):
    # A dedent to a width that is not on the indent stack (2.6): after a line
    # at width 8, the next line at width 4 or less is moved to width 6.
    ls = s.split("\n")
    w = lambda l: len(l) - len(l.lstrip(" "))
    idx = [i for i in range(len(ls) - 1) if ls[i].strip() and w(ls[i]) == 8 and ls[i + 1].strip()
           and w(ls[i + 1]) == 4 and not ls[i + 1].lstrip().startswith("#")]
    if not idx: return None
    i = rnd.choice(idx) + 1; ls[i] = "      " + ls[i].lstrip(" "); return "\n".join(ls)
def mut_bigint(s): return s + "\nconst BIG_LIT = 99999999999999999999999\n"
def mut_badu(s): return s + '\nconst BAD_U = "\\u{110000}"\n'
def mut_unsafe(s): return s + "\nfn unsafe_probe() -> i64\n    let unsafe = 1\n    return unsafe\n"
def mut_unknown(s): return s + "\nfn probe_unknown() -> i64\n    return no_such_name_zz\n"
def mut_dup(s): return s + "\nfn dup_zz() -> i64\n    return 1\n\nfn dup_zz() -> i64\n    return 2\n"
def mut_paren(s): return s + "\nfn paren_zz() -> i64\n    return (1 + 2\n"
MUTS = [("W-LEX-1", mut_tab, "tab"), ("W-LEX-1", mut_cr, "cr"), ("W-LEX-1", mut_bom, "bom"), ("W-LEX-2", mut_dedent, "dedent"),
        ("W-LEX-3", mut_bigint, "bigint"), ("W-LEX-3", mut_badu, "bad-escape"),
        ("W-SAFE-1", mut_unsafe, "unsafe"), ("W-NAME-1", mut_unknown, "unknown-name"), ("W-NAME-2", mut_dup, "duplicate"),
        ("W-SYN-1", mut_paren, "paren")]
for code, f, tag in MUTS:
    for stem, src in rnd.sample(accepted, min(12, len(accepted))):
        m = f(src)
        if m is None or m == src: continue
        r = ref.check(m)
        got = [d.get("code") for d in r.get("diagnostics", [])] if r.get("verdict") == "fit" else ["ok"]
        if code not in got:
            findings.append({"kind": "lex-mutation", "intended": code, "reference": got[:3], "mutation": tag, "base": stem})
        area = "lexing-parsing" if code.startswith(("W-LEX", "W-SYN", "W-VER")) else "types-checking"
        put(area, "generated", f"mut-{tag}-{stem}", m, {"clauses": [code], "expect": "fit", "diagnostic": code,
            "provenance": f"gen_props.py seed {seed}: mutation `{tag}` of {stem}"})

# ---- T7 name independence on traces. Only names that do not reach D are
# renamed (functions and locals). Event and error names are written into D
# as data ("type"), so renaming them changes D's content; a hash injective on
# D must then change too (see finding T7-SCOPE). The test keeps that law.
for p in sorted(out.rglob("*-trace.rce")):
    src, meta = rcetest.split(p.read_text(encoding="utf-8"))
    names = set(re.findall(r"^fn (\w+)", src, re.M)) | set(re.findall(r"\b(?:let|var) (\w+)", src))
    names -= {"main", "_"}
    if not names: continue
    ren = {n: f"zq_{n}_renamed" for n in names}
    rx = re.compile(r"(?<![.\w])(" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")\b")
    s2 = rx.sub(lambda m: ren[m.group(1)], src)
    if ref.check(s2).get("verdict") != "ok": continue   # a name also used as a field; skip
    r = ref.trace(s2, meta["events"])
    got = (r.get("trace") or {}).get("head")
    if got != meta["trace_head"]:
        findings.append({"kind": "T7", "base": str(p), "renamed_head": got, "original_head": meta["trace_head"],
                         "error": r.get("error")})
    put("forms-log", "generated", f"t7-{p.stem}", s2, {"clauses": ["T7", "T3"], "expect": "trace", "events": meta["events"],
        "trace_head": meta["trace_head"], "provenance": f"gen_props.py: functions and locals of {p.name} renamed; head must not change (T7)"})

# Event and error renaming, recorded once as evidence for finding T7-SCOPE.
for p in sorted(out.rglob("01-counter-ex2-trace.rce")):
    src, meta = rcetest.split(p.read_text(encoding="utf-8"))
    s2 = re.sub(r"\bInc\b", "Up", src); r = ref.trace(s2, re.sub(r"\bInc\b", "Up", meta["events"]))
    findings.append({"kind": "T7-SCOPE", "base": str(p), "original_head": meta["trace_head"],
                     "renamed_head": (r.get("trace") or {}).get("head"),
                     "note": "event rename Inc->Up changes every log hash; T7 as written says it must not"})

# ---- T16 par schedule independence, W-PAR-1
for k in range(40):
    nb = rnd.randint(2, 4); vars_ = [f"v{i}" for i in range(nb)]
    init = [rnd.randint(-50, 50) for _ in vars_]; body = ["# rce-version: 0.6", "fn main()"]
    for v, x in zip(vars_, init): body.append(f"    var {v}: i64 = {x}")
    body.append("    var acc: List[i64] = [" + ", ".join(str(rnd.randint(0, 9)) for _ in range(5)) + "]")
    body.append("    par")
    for i, v in enumerate(vars_):
        body.append(f"        branch b{i}")
        body.append(f"            for x in acc")
        body.append(f"                {v} = {v} {rnd.choice(['+', '-'])} x * {rnd.randint(1, 5)}")
    for v in vars_: body.append(f"    emit {v}")
    src = "\n".join(body) + "\n"
    r = ref.run(src)
    if r.get("verdict") != "ok" or "error" in r:
        findings.append({"kind": "par-generator", "src": src, "reference": r}); continue
    put("concurrency", "generated", f"par-{seed}-{k:03d}", src, {"clauses": ["T16", "W-PAR-1"], "expect": "run",
        "schedule_seeds": [1, 2, 3, 4, 5], "outcome": r["outcome"], "outputs": r["outputs"],
        "provenance": f"gen_props.py seed {seed}: disjoint footprints, any schedule gives the source-order result"})
    # overlap: branch 1 also writes branch 0's variable
    bad = src.replace(f"                {vars_[1]} = {vars_[1]}", f"                {vars_[0]} = {vars_[0]}", 1)
    rb = ref.check(bad)
    codes = [d.get("code") for d in rb.get("diagnostics", [])]
    if "W-PAR-1" not in codes:
        findings.append({"kind": "W-PAR-1", "src": bad, "reference": codes or rb.get("verdict")})
    put("concurrency", "generated", f"par-overlap-{seed}-{k:03d}", bad, {"clauses": ["W-PAR-1"], "expect": "fit",
        "diagnostic": "W-PAR-1", "provenance": "gen_props.py: two branches write one variable (write/write overlap)"})

# ---- HC-15 / HC-16 admission order independent of arrival
prio = (pathlib.Path(os.environ["RCE_SPEC"]) / "checker/tests/host/priority.rce").read_text()
def offers():
    v = []
    for t in range(12):
        v.append({"source": "kbd", "seq": t + 1, "tick": t, "event": "Key", "args": {"c": chr(97 + t)}})
        if t % 3 == 0: v.append({"source": "irq", "seq": t // 3 + 1, "tick": t, "event": "Alarm", "args": {"n": t // 3 + 1}})
        if t % 2 == 1: v.append({"source": "net", "seq": t // 2 + 1, "tick": t, "event": "Key", "args": {"c": chr(65 + t)}})
    return v
def steps_for(order, retransmit):
    st = []
    for w, upto in enumerate([3, 7, 11]):
        xs = [o for o in order if (o["tick"] <= 3 if w == 0 else (3 < o["tick"] <= 7 if w == 1 else o["tick"] > 7))]
        for i, x in enumerate(xs):
            st.append({"offer": x})
            if retransmit and i % retransmit == 0: st.append({"offer": x})
        st.append({"close": upto})
    return st
base = ref.call(op="host", source=prio, steps=steps_for(offers(), 0))
want = {k: base["host"][k] for k in ("state_hash", "admitted")}
want["entries"] = [e for e in base["host"]["entries"] if e not in ("host.FabricCaps", "host.StorageCaps")]
for k in range(25):
    o = offers(); rnd.shuffle(o); rt = rnd.choice([0, 1, 2, 3])
    st = steps_for(o, rt)
    r = ref.call(op="host", source=prio, steps=st)
    h = r.get("host", {})
    if h.get("state_hash") != want["state_hash"] or h.get("admitted") != want["admitted"]:
        findings.append({"kind": "HC-15", "case": k, "reference": h.get("state_hash"), "law": want["state_hash"]})
    put("host-contract", "generated", f"admit-order-{seed}-{k:03d}", prio, {"clauses": ["HC-15", "HC-16", "HC-17", "HC-9", "HC-8", "T25"],
        "expect": "host", "steps": st, "host_expect": want,
        "provenance": f"gen_props.py seed {seed}: arrival permutation {k}, retransmit every {rt or 'never'}; D must not depend on arrival"})

# ---- fuel boundary
for p in sorted(out.rglob("*-main.rce")):
    src, meta = rcetest.split(p.read_text(encoding="utf-8"))
    full = ref.run(src)
    if full.get("outcome") != "done": continue
    lo, hi = 0, 1
    while ref.run(src, fuel=hi).get("outcome") != "done" and hi < 2**40: hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if ref.run(src, fuel=mid).get("outcome") == "done": hi = mid
        else: lo = mid
    at, below = ref.run(src, fuel=hi), ref.run(src, fuel=hi - 1)
    if at.get("outputs") != full.get("outputs"):
        findings.append({"kind": "fuel-changes-value", "base": str(p), "fuel": hi})
    put("evaluation-fuel", "generated", f"fuel-ok-{p.stem}", src, {"clauses": ["T2", "HC-7"], "expect": "run", "fuel": hi,
        "outcome": "done", "outputs": full["outputs"], "provenance": f"gen_props.py: least fuel that completes {p.name}"})
    put("evaluation-fuel", "generated", f"fuel-trap-{p.stem}", src, {"clauses": ["T2", "HC-7"], "expect": "run", "fuel": hi - 1,
        "outcome": below.get("outcome"), "provenance": f"gen_props.py: one below the least fuel of {p.name}"})

os.makedirs(out.parent / "findings", exist_ok=True)
json.dump(findings, open(out.parent / "findings" / f"props-{seed}.json", "w"), indent=1)
print(json.dumps({"written": count, "findings": len(findings)}))
