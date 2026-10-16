#!/usr/bin/env python3
"""Count every figure the prose quotes from the source, and check the prose.

  scripts/claim-counts.py            print the counts as JSON
  scripts/claim-counts.py --check    fail when a quoted figure disagrees with the source

A count is a count: this script never reports that a test passes. Pass/fail
needs a run; the only run it uses is `rce-spec-check` (the grammar and
example numbers), found at $RCE_SPEC_CHECK or built with
  cd spec/checker && cargo build --release --bin rce-spec-check
Lean: the module count is a file count; the audit count needs
`scripts/lean-audit.sh` and is read from $RCE_LEAN_AUDIT_OUT if set
(the text printed by `lake env lean audit/Audit.lean`).
"""
import json, os, pathlib, re, subprocess, sys

root = pathlib.Path(__file__).resolve().parent.parent
spec = root / "spec"
book = spec / "book"


def read(p):
    return (root / p).read_text()


def spec_check_output():
    exe = os.environ.get("RCE_SPEC_CHECK")
    if not exe:
        for cand in (spec / "checker/target/release/rce-spec-check",):
            if cand.exists():
                exe = str(cand)
    if not exe:
        return None
    r = subprocess.run([exe], cwd=spec, capture_output=True, text=True)
    return r.stdout + r.stderr


def counts():
    c = {}
    ex = spec / "examples"
    c["examples_files"] = len(list(ex.glob("*.rce")))
    c["examples_package_files"] = sum(len(list(d.rglob("*.rce"))) for d in ex.iterdir()
                                      if d.is_dir() and d.name not in ("reject", "ill_typed", "inadmissible"))
    c["examples_total"] = c["examples_files"] + c["examples_package_files"]
    for d in ("reject", "ill_typed", "inadmissible"):
        c["examples_" + d] = len(list((ex / d).glob("*.rce")))
    cli = read("spec/checker/src/cli.rs")
    m = re.search(r'"rust" \| "python" \| "js" \| "scala" =>', cli)
    c["emit_targets"] = re.findall(r'"(\w+)"', m.group(0)) if m else None
    src = "".join(p.read_text(errors="ignore") for p in (spec / "checker/src").rglob("*.rs"))
    c["checker_src_mentions_haskell"] = len(re.findall(r"haskell", src, re.I))
    c["checker_src_mentions_julia"] = len(re.findall(r"julia", src, re.I))
    c["versions"] = re.findall(r'"(0\.\d)"', re.search(r"pub const VERSIONS: &\[&str\] = &\[(.*?)\];", read("spec/checker/src/parse.rs")).group(1))
    tests = list((spec / "checker/tests").rglob("*.rs"))
    c["checker_test_files"] = len(tests)
    c["checker_test_fns"] = sum(len(re.findall(r"#\[test\]", p.read_text(errors="ignore"))) for p in tests)
    c["lean_modules"] = len(list((spec / "lean/RCE").glob("*.lean")))
    th = read("spec/book/10-theorems.md")
    c["theorem_max_id"] = max(int(x) for x in re.findall(r"\bT(\d+)\b", th))
    vp = spec / "data/canonical-json-vectors.json"
    if vp.exists():
        c["canonical_json_vectors"] = len(json.loads(vp.read_text())["vectors"])
    out = spec_check_output()
    if out:
        g = re.search(r"LALR\(1\) from book/03-syntax.md: (\d+) rules, (\d+) productions, (\d+) states, (\d+) conflicts", out)
        h = re.search(r"versions 0\.6 and 0\.7 \(section 3 with C\.2\): (\d+) rules, (\d+) productions, (\d+) states, (\d+) conflicts", out)
        a = re.search(r"grammar: accepted (\d+)/(\d+) valid examples, refused (\d+)/(\d+) invalid", out)
        k = re.search(r"checker: accepted (\d+)/(\d+) examples, refused (\d+)/(\d+) syntax rejects, refused (\d+)/(\d+) ill-typed or inadmissible \((\d+) of them inadmissible", out)
        c["spec_check"] = {
            "all_ok": "ALL OK" in out,
            "grammar05": list(map(int, g.groups())) if g else None,
            "grammar067": list(map(int, h.groups())) if h else None,
            "grammar_examples": list(map(int, a.groups())) if a else None,
            "checker": list(map(int, k.groups())) if k else None,
        }
    aud = os.environ.get("RCE_LEAN_AUDIT_OUT")
    if aud and pathlib.Path(aud).exists():
        mm = re.search(r"lean audit: (\d+) RCE theorems", pathlib.Path(aud).read_text())
        c["lean_audit_theorems"] = int(mm.group(1)) if mm else None
    return c


def need(cond, msg, errs):
    if not cond:
        errs.append(msg)


def check(c):
    errs = []
    sc = c.get("spec_check")
    need(sc is not None, "rce-spec-check not built: set RCE_SPEC_CHECK", errs)
    ov, syn, tool = read("spec/book/00-overview.md"), read("spec/book/03-syntax.md"), read("spec/book/12-tooling.md")
    if sc:
        r, p, s, cf = sc["grammar05"][0], sc["grammar05"][1], sc["grammar05"][2], sc["grammar05"][3]
        for name, txt in (("00-overview", ov), ("03-syntax", syn)):
            need(f"{r} rules, {p} productions, {s} states" in re.sub(r"\s+", " ", txt)
                 or f"{r} rules, {p} productions, {s} states" in re.sub(r"\s+", " ", txt), f"{name}: grammar numbers != {sc['grammar05']}", errs)
        o = re.sub(r"\s+", " ", ov)
        need(f"{sc['grammar067'][0]} rules, {sc['grammar067'][1]} productions, {sc['grammar067'][2]} states" in o, "00-overview: 0.6/0.7 grammar numbers", errs)
        ga, ck = sc["grammar_examples"], sc["checker"]
        need(f"grammar {ga[0]}/{ga[1]} accepted, {ga[2]}/{ga[3]} refused" in o, "00-overview: grammar accepted/refused", errs)
        need(f"checker {ck[0]}/{ck[1]} accepted, {ck[2]}/{ck[3]} and {ck[4]}/{ck[5]} refused ({ck[6]} of them inadmissible)" in o, "00-overview: checker line", errs)
        need(ga[1] == c["examples_total"], "spec-check example count != files counted", errs)
        need(ck[5] == c["examples_ill_typed"] + c["examples_inadmissible"], "ill_typed+inadmissible != 64", errs)
    t = re.sub(r"\s+", " ", tool)
    need(f"emits all {c['examples_total']} programs of `spec/examples/`" in t, "12-tooling: emitter corpus size", errs)
    s3 = re.sub(r"\s+", " ", syn)
    need(f"parses the {c['examples_total']} programs in `examples/`" in s3, "03-syntax: example count", errs)
    # emit targets named by the docs equal those the code accepts
    readme = read("README.md")
    tg = c["emit_targets"]
    need(tg == ["rust", "python", "js", "scala"], f"cli emit targets changed: {tg}", errs)
    need(c["checker_src_mentions_haskell"] == 0 and c["checker_src_mentions_julia"] == 0,
         "checker source now mentions Haskell or Julia: review README and 12-tooling", errs)
    for bad in ("Haskell emit", "Julia probe", "IDE, 811", "projects the program to Rust, Scala, Haskell"):
        need(bad not in readme and bad not in tool, f"unreproducible claim still present: {bad!r}", errs)
    # versions
    w = re.sub(r"\s+", " ", read("spec/book/09-wellformedness.md"))
    vs = c["versions"]
    need(f"other than {', '.join(vs[:-1])} or {vs[-1]} (2.7)" in w, "09: W-VER-1 version list != parse.rs VERSIONS", errs)
    lex = re.sub(r"\s+", " ", read("spec/book/02-lexical.md"))
    need(", ".join(f"`{v}`" for v in vs[:-1]) + f" and `{vs[-1]}`; any other is refused" in lex, "02: 2.7 version list != parse.rs VERSIONS", errs)
    # canonical json section
    cj = re.sub(r"\s+", " ", read("spec/book/07-form-evaluation.md"))
    if "canonical_json_vectors" in c:
        need(f"holds {c['canonical_json_vectors']} vectors" in cj, "07: vector count in 6.1.1", errs)
    # decisions that were stale
    dec = read("spec/book/11-decisions.md")
    need("Grammar artifact is Lark" not in dec, "11: decision 29 still names Lark as the artifact", errs)
    need("Current result (draft 0.5)" not in ov, "00: overview still says draft 0.5 result", errs)
    return errs


def main():
    c = counts()
    if "--check" in sys.argv:
        errs = check(c)
        for e in errs:
            print("MISMATCH:", e)
        print("claim-counts: %s" % ("FAIL" if errs else "all quoted figures match the source"))
        sys.exit(1 if errs else 0)
    print(json.dumps(c, indent=1))


main()
