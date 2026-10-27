#!/usr/bin/env python3
"""Count every figure the prose quotes from the source, check the prose, and
write the facts file.

  scripts/claim-counts.py                      print the counts as JSON
  scripts/claim-counts.py --check              fail when a quoted figure disagrees with the source
  scripts/claim-counts.py --runs DIR --write PATH
                                               write the facts file (counts and run outcomes)
  scripts/claim-counts.py --check --runs DIR --facts PATH
                                               also require PATH to equal a fresh recomputation

A count is a count: the counts read the source and never report that a test
passes. Run outcomes are a second, separate input: the logs and exit statuses
that scripts/verify-run.sh writes to DIR (tests, spec_check,
spec_check_three_parsers, wasm, and lean or lean-audit.ci.log). The facts
file keeps the two apart: summary.counts from the source, summary.runs from
the logs, each entry with its definition and its evidence.

Without --runs, the specification-check numbers come from `rce-spec-check`,
found at $RCE_SPEC_CHECK or built with
  cd spec/checker && cargo build --release --bin rce-spec-check
and the Lean audit count from $RCE_LEAN_AUDIT_OUT if set (the text printed by
`lake env lean audit/Audit.lean`).

Options: --base REV records whether the tested trees equal those of REV (a
commit of main); --site-facts PATH carries the site's own keys
(spec_notebook_draft, spec_notebook_commit) from an earlier facts file.
"""
import hashlib, json, os, pathlib, re, shutil, subprocess, sys

root = pathlib.Path(__file__).resolve().parent.parent
spec = root / "spec"
book = spec / "book"

SCHEMA = "rce-facts/2"
REPO = "https://github.com/openIE-dev/rce"
# trees whose content the checks read; a run commit off main is a run of main
# only when every one of them equals the tree at that main commit
TESTED_TREES = ["spec/checker", "spec/lean", "spec/examples", "spec/conc06", "spec/conc06b",
                "spec/data", "spec/book", "spec/checker-wasm", "tree-sitter"]
SITE_KEYS = ["spec_notebook_draft", "spec_notebook_commit"]
EXTERNAL_TOOLS = ["rustc", "python3", "node", "tree-sitter", "scala-cli", "lake", "java", "uv",
                  "sqlite3", "timeout", "go"]


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


def parse_spec_check_compat(out):
    g = re.search(r"LALR\(1\) from book/03-syntax.md: (\d+) rules, (\d+) productions, (\d+) states, (\d+) conflicts", out)
    h = re.search(r"section 3 with C\.2: (\d+) rules, (\d+) productions, (\d+) states, (\d+) conflicts", out)
    a = re.search(r"grammar: accepted (\d+)/(\d+) valid examples, refused (\d+)/(\d+) invalid", out)
    k = re.search(r"checker: accepted (\d+)/(\d+) examples, refused (\d+)/(\d+) syntax rejects, refused (\d+)/(\d+) ill-typed or inadmissible \((\d+) of them inadmissible", out)
    return {
        "all_ok": "ALL OK" in out,
        "grammar05": list(map(int, g.groups())) if g else None,
        "grammar067": list(map(int, h.groups())) if h else None,
        "grammar_examples": list(map(int, a.groups())) if a else None,
        "checker": list(map(int, k.groups())) if k else None,
    }


def counts(spec_out=None):
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
    tests = list((spec / "checker/tests").rglob("*.rs"))
    c["checker_test_files"] = len(tests)
    c["checker_test_fns"] = sum(len(re.findall(r"#\[test\]", p.read_text(errors="ignore"))) for p in tests)
    c["lean_modules"] = len(list((spec / "lean/RCE").glob("*.lean")))
    th = read("spec/book/10-theorems.md")
    c["theorem_max_id"] = max(int(x) for x in re.findall(r"\bT(\d+)\b", th))
    vp = spec / "data/canonical-json-vectors.json"
    if vp.exists():
        c["canonical_json_vectors"] = len(json.loads(vp.read_text())["vectors"])
    out = spec_out if spec_out is not None else spec_check_output()
    if out:
        c["spec_check"] = parse_spec_check_compat(out)
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
    w = re.sub(r"\s+", " ", read("spec/book/09-wellformedness.md"))
    # one language: the release label is tolerated, so no version list is claimed in 02 or 09
    lex = re.sub(r"\s+", " ", read("spec/book/02-lexical.md"))
    need("A release label names exactly one spec commit" in lex, "02: 2.7 must state that a release label names exactly one spec commit", errs)
    need("W-VER-1" not in w and "W-VER-2" not in w, "09: W-VER rules no longer exist", errs)
    # canonical json section
    cj = re.sub(r"\s+", " ", read("spec/book/07-form-evaluation.md"))
    if "canonical_json_vectors" in c:
        need(f"holds {c['canonical_json_vectors']} vectors" in cj, "07: vector count in 6.1.1", errs)
    # decisions that were stale
    dec = read("spec/book/11-decisions.md")
    need("Grammar artifact is Lark" not in dec, "11: decision 29 still names Lark as the artifact", errs)
    need("Current result (draft 0.5)" not in ov, "00: overview still says draft 0.5 result", errs)
    return errs


# ---------------------------------------------------------------- source counts

def strip_lean(text, strings=False):
    """Remove Lean line comments and nested block comments (doc comments
    included); with strings=True also remove string literals. Newlines are
    kept, so line structure is unchanged."""
    out, i, n, depth, in_str = [], 0, len(text), 0, False
    while i < n:
        ch = text[i]
        if in_str:
            if ch == "\\" and i + 1 < n:
                if not strings:
                    out.append(text[i:i + 2])
                i += 2
                continue
            if ch == '"':
                in_str = False
            if not strings or ch == "\n":
                out.append(ch)
            i += 1
            continue
        if depth == 0 and text.startswith("--", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("/-", i):
            depth += 1
            i += 2
            continue
        if depth > 0:
            if text.startswith("-/", i):
                depth -= 1
                i += 2
                continue
            if ch == "\n":
                out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_str = True
            if not strings:
                out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


LEAN_PREFIX = r"^[ \t]*(?:@\[[^\]\n]*\][ \t]*)*(?:(?:private|protected|noncomputable|nonrec|partial|unsafe)[ \t]+)*"
LEAN_DECL = re.compile(LEAN_PREFIX + r"(theorem|lemma)\b", re.M)
LEAN_AXIOM = re.compile(LEAN_PREFIX + r"axiom\b", re.M)


def lean_counts():
    lean = spec / "lean"
    files = sorted(p for p in lean.rglob("*.lean") if ".lake" not in p.parts)
    th = le = ax = so = 0
    for p in files:
        t = p.read_text(errors="ignore")
        s = strip_lean(t)
        for m in LEAN_DECL.finditer(s):
            if m.group(1) == "theorem":
                th += 1
            else:
                le += 1
        ax += len(LEAN_AXIOM.findall(s))
        so += len(re.findall(r"\bsorry\b", strip_lean(t, strings=True)))
    return {
        "lean_files_top": len(list((lean / "RCE").glob("*.lean"))),
        "lean_files_rce_all": len([p for p in files if (lean / "RCE") in p.parents]),
        "lean_files_all": len(files),
        "lean_theorem_declarations": th,
        "lean_lemma_declarations": le,
        "lean_declarations": th + le,
        "lean_axiom_declarations": ax,
        "lean_sorry_in_source": so,
    }


def test_attribute_counts():
    pat = re.compile(r"^\s*#\[test\]", re.M)
    def count(d):
        return sum(len(pat.findall(p.read_text(errors="ignore"))) for p in sorted((spec / d).rglob("*.rs")))
    t, s = count("checker/tests"), count("checker/src")
    return {"checker_test_attributes_tests": t, "checker_test_attributes_src": s,
            "checker_test_attributes": t + s,
            "checker_test_files": len(list((spec / "checker/tests").rglob("*.rs")))}


def spec_document_draft():
    m = re.search(r"^# RCE Language Specification, draft ([0-9.]+)", read("spec/book/00-overview.md"), re.M)
    return m.group(1) if m else None


DEF = {
    "examples_top_level": "files spec/examples/*.rce",
    "examples_in_packages": "files spec/examples/<directory>/**/*.rce, for every directory other than reject, ill_typed and inadmissible",
    "examples_valid_total": "examples_top_level + examples_in_packages",
    "examples_dir_reject": "files spec/examples/reject/*.rce: programs whose expected outcome is a syntax diagnostic",
    "examples_dir_ill_typed": "files spec/examples/ill_typed/*.rce: programs whose expected outcome is a type diagnostic",
    "examples_dir_inadmissible": "files spec/examples/inadmissible/*.rce: programs whose expected outcome is an admissibility diagnostic",
    "canonical_json_vectors": "length of the array vectors in spec/data/canonical-json-vectors.json",
    "emit_targets": "target names that `rce emit` dispatches to a program emitter in spec/checker/src/cli.rs",
    "emit_form": "true when `rce emit form` (the Form JSON) is dispatched in spec/checker/src/cli.rs",
    "theorem_max_id": "the largest n of a label Tn in spec/book/10-theorems.md",
    "spec_document_draft": "the draft label in the title of spec/book/00-overview.md",
    "checker_test_files": "files spec/checker/tests/**/*.rs",
    "checker_test_attributes_tests": "lines matching ^\\s*#\\[test\\] in spec/checker/tests/**/*.rs",
    "checker_test_attributes_src": "lines matching ^\\s*#\\[test\\] in spec/checker/src/**/*.rs",
    "checker_test_attributes": "checker_test_attributes_tests + checker_test_attributes_src",
    "lean_files_top": "files spec/lean/RCE/*.lean (one directory level)",
    "lean_files_rce_all": "files spec/lean/RCE/**/*.lean (all subdirectories)",
    "lean_files_all": "every .lean file under spec/lean/ (outside .lake)",
    "lean_theorem_declarations": "in every .lean file under spec/lean/, after removing line comments and nested block comments, the lines whose first word is theorem, optionally after attributes @[...] and the modifiers private, protected, noncomputable, nonrec, partial or unsafe",
    "lean_lemma_declarations": "as lean_theorem_declarations, with the word lemma",
    "lean_declarations": "lean_theorem_declarations + lean_lemma_declarations",
    "lean_axiom_declarations": "as lean_theorem_declarations, with the word axiom",
    "lean_sorry_in_source": "occurrences of the word sorry in every .lean file under spec/lean/ after removing comments and string literals",
}


def source_counts():
    c = counts(spec_out="")
    cli = read("spec/checker/src/cli.rs")
    v = {
        "examples_top_level": c["examples_files"],
        "examples_in_packages": c["examples_package_files"],
        "examples_valid_total": c["examples_total"],
        "examples_dir_reject": c["examples_reject"],
        "examples_dir_ill_typed": c["examples_ill_typed"],
        "examples_dir_inadmissible": c["examples_inadmissible"],
        "canonical_json_vectors": c.get("canonical_json_vectors"),
        "emit_targets": c["emit_targets"],
        "emit_form": bool(re.search(r'^\s*"form" =>', cli, re.M)),
        "theorem_max_id": c["theorem_max_id"],
        "spec_document_draft": spec_document_draft(),
    }
    v.update(test_attribute_counts())
    v.update(lean_counts())
    return {k: {"value": v[k], "definition": DEF[k]} for k in DEF}


# ---------------------------------------------------------------- run logs

def run_file(runs, name):
    p = pathlib.Path(runs) / name
    return p.read_text(errors="replace") if p.exists() else None


def run_status(runs, step):
    s = run_file(runs, step + ".status")
    return int(s.strip()) if s and s.strip().lstrip("-").isdigit() else None


def parse_cargo_test(text):
    targets, cur = [], None
    fails, ignored = [], []
    for line in text.splitlines():
        m = re.match(r"\s*Running (?:unittests )?(\S+)", line)
        d = re.match(r"\s*Doc-tests (\S+)", line)
        if m or d:
            cur = {"target": ("doc-tests " + d.group(1)) if d else m.group(1)}
            targets.append(cur)
            continue
        r = re.match(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out", line)
        if r and cur is not None and "passed" not in cur:
            cur.update(result=r.group(1), passed=int(r.group(2)), failed=int(r.group(3)),
                       ignored=int(r.group(4)), measured=int(r.group(5)), filtered_out=int(r.group(6)))
            continue
        t = re.match(r"test (\S+) \.\.\. (FAILED|ignored)", line)
        if t:
            (fails if t.group(2) == "FAILED" else ignored).append(t.group(1))
    tot = {k: sum(x.get(k, 0) for x in targets) for k in ("passed", "failed", "ignored", "measured", "filtered_out")}
    tot["targets"] = len(targets)
    tot["targets_without_result"] = [x["target"] for x in targets if "passed" not in x]
    tot["failed_tests"] = sorted(fails)
    tot["ignored_tests"] = sorted(ignored)
    tot["per_target"] = targets
    return tot


def parse_spec_check(text):
    lines = [l for l in text.splitlines() if re.match(r"(grammar|checker|book|three|tree-sitter|parsers?)\b|ALL OK", l)]
    lalr = [{"grammar": m.group(1), "rules": int(m.group(2)), "productions": int(m.group(3)),
             "states": int(m.group(4)), "conflicts": int(m.group(5))}
            for m in re.finditer(r"LALR\(1\) (?:from|of) (.+?): (\d+) rules, (\d+) productions, (\d+) states, (\d+) conflicts", text)]
    ts = re.search(r"grammar: tree-sitter LR\(1\), (\d+) conflicts, (\d+) rules", text)
    a = re.search(r"grammar: accepted (\d+)/(\d+) valid examples, refused (\d+)/(\d+) invalid(?: \((.*)\))?$", text, re.M)
    k = re.search(r"checker: accepted (\d+)/(\d+) examples, refused (\d+)/(\d+) syntax rejects, refused (\d+)/(\d+) ill-typed or inadmissible \((\d+) of them inadmissible", text)
    tree_sitter = {"rules": int(ts.group(2)), "conflicts": int(ts.group(1))} if ts else None
    return {
        "all_ok": bool(re.search(r"^ALL OK$", text, re.M)),
        "lalr_tables": lalr,
        "tree_sitter_lr1": tree_sitter,
        "grammar_conflicts_total": sum(x["conflicts"] for x in lalr) + (tree_sitter["conflicts"] if tree_sitter else 0),
        "grammar_examples": {"valid_parsed": [int(a.group(1)), int(a.group(2))],
                             "dir_reject_diagnosed": [int(a.group(3)), int(a.group(4))],
                             "parsers": a.group(5)} if a else None,
        "checker_examples": {"valid_checked": [int(k.group(1)), int(k.group(2))],
                             "dir_reject_diagnosed": [int(k.group(3)), int(k.group(4))],
                             "ill_typed_or_inadmissible_diagnosed": [int(k.group(5)), int(k.group(6))],
                             "of_them_inadmissible": int(k.group(7))} if k else None,
        "lines": lines,
    }


def parse_wasm(text):
    t = re.search(r"^(\d+)/(\d+) wasm checks passed$", text, re.M)
    b = re.search(r"wasm checks by component: grammar \(tree-sitter-rce\.wasm\) (\d+)/(\d+), checker \(checker-wasm/pkg\) (\d+)/(\d+)", text)
    return {
        "total": [int(t.group(1)), int(t.group(2))] if t else None,
        "grammar": [int(b.group(1)), int(b.group(2))] if b else None,
        "checker": [int(b.group(3)), int(b.group(4))] if b else None,
        "fail_lines": len(re.findall(r"^FAIL ", text, re.M)),
        "lines": [l for l in text.splitlines() if "wasm checks" in l or l.startswith("FAIL ")],
    }


def strip_ci_prefix(line):
    return re.sub(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ", "", line)


def parse_lean(text, ci):
    lines = [strip_ci_prefix(l) for l in text.splitlines()]
    aud = [l for l in lines if re.match(r"lean audit: ", l)]
    m = None
    for l in aud:
        m = re.match(r"lean audit: (\d+) RCE theorems, standard axioms only$", l) or m
    sorry = [l for l in lines if re.search(r"\b(warning|error)\b", l) and re.search(r"\bsorry\b", l)]
    r = {
        "audit_line": aud[-1] if aud else None,
        "audit_theorem_constants": int(m.group(1)) if m else None,
        "standard_axioms_only": bool(m),
        "sorry_messages": len(sorry),
        "build_completed": any(l.strip() == "Build completed successfully." for l in lines),
        "lake_built_lines": len([l for l in lines if re.search(r"\bBuilt \S", l)]),
        "lean_version": next((l.strip() for l in lines if l.strip().startswith("Lean (version ")), None),
    }
    if ci:
        for i, l in enumerate(lines):
            if l.endswith("git log -1 --format=%H") and i + 1 < len(lines):
                h = lines[i + 1].strip()
                if re.fullmatch(r"[0-9a-f]{40}", h):
                    r["commit"] = h
        r["lake_cache_restored"] = not any(l.startswith("Cache not found for input keys") for l in lines)
    return r


def git(*a):
    try:
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def tool_version(cmd, cwd=None):
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (r.stdout + r.stderr).strip().splitlines()
    out = [l for l in out if not l.startswith("Picked up JAVA_TOOL_OPTIONS")]
    return out[0] if out and r.returncode == 0 else None


def toolchains():
    lock = spec / "wasm-test/package-lock.json"
    wts = None
    if lock.exists():
        wts = json.loads(lock.read_text()).get("packages", {}).get("node_modules/web-tree-sitter", {}).get("version")
    rt = re.search(r'channel\s*=\s*"([^"]+)"', read("spec/checker/rust-toolchain.toml"))
    return {
        "rust_pinned": rt.group(1) if rt else None,
        "rustc": tool_version(["rustc", "--version"], cwd=spec / "checker"),
        "cargo": tool_version(["cargo", "--version"], cwd=spec / "checker"),
        "node": tool_version(["node", "--version"]),
        "web_tree_sitter_pinned": wts,
        "tree_sitter_cli": tool_version(["tree-sitter", "--version"]),
        "python": "Python " + sys.version.split()[0],
        "lean_pinned": read("spec/lean/lean-toolchain").strip(),
        "external_tools_on_path": {t: shutil.which(t) is not None for t in EXTERNAL_TOOLS},
    }


def sha256_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def wasm_builds():
    out = {}
    for name, rel in (("checker", "spec/checker-wasm/pkg/rce_spec_check_wasm_bg.wasm"),
                      ("grammar", "tree-sitter/tree-sitter-rce.wasm")):
        last = git("log", "-1", "--format=%H", "--", rel)
        out[name] = {"path": rel, "sha256": sha256_file(root / rel), "last_changed_in": last or None}
    last = out["checker"]["last_changed_in"]
    n = git("rev-list", "--count", f"{last}..HEAD", "--", "spec/checker/src") if last else None
    out["checker"]["commits_changing_spec_checker_src_since"] = int(n) if n and n.isdigit() else None
    return out


RUN_DEF = {
    "tests": "the test binaries that cargo test --release builds in spec/checker (default features), run by scripts/verify-tests.py tests from hard links to the files of one build, with the target directory's lock held while a target that runs target/release/rce runs, and TMPDIR private to the run; doc tests through cargo test --release --doc. Sums over every 'test result:' line; a test that returns early because an external tool is absent is reported as passed (see toolchains.external_tools_on_path)",
    "spec_check": "rce-spec-check built with cargo build --release and run in spec/checker by scripts/verify-tests.py bin, with the target directory's lock held",
    "spec_check_three_parsers": "the same with --features tree-sitter",
    "wasm": "npm ci && node test.mjs in spec/wasm-test, over the committed builds named in wasm_builds; a check whose label begins with 'ts' is a grammar check, every other check a checker check",
    "lean": "lake build of spec/lean and the axiom audit spec/lean/audit/Audit.lean (scripts/lean-audit.sh, or the CI job of .github/workflows/lean.yml that runs the same two commands): every theorem constant of a module under RCE uses only propext, Quot.sound and Classical.choice; sorryAx is not among them",
}


def outcome(ok, ran):
    return "not run" if not ran else ("passed" if ok else "did not pass")


def runs_summary(runs):
    res = {}
    t = run_file(runs, "tests.log")
    if t is not None:
        p = parse_cargo_test(t)
        st = run_status(runs, "tests")
        b = json.loads(run_file(runs, "tests.binaries.json") or "null")
        iso = None
        if b:
            iso = {"executables": b["executables"], "targets_run_under_lock": b["targets_run_under_lock"],
                   "rce_binary_embeds_run_commit": b["rce_binary"]["embeds_run_commit"],
                   "rce_binary_unchanged_while_locked": b["rce_binary"]["unchanged_while_locked"],
                   "hard_links_unchanged_at_end": b["hard_links_unchanged_at_end"],
                   "binary_hashes": "tests.binaries.json of the run (host-dependent, so outside the summary)"}
        iso_ok = bool(iso and iso["rce_binary_embeds_run_commit"] and iso["rce_binary_unchanged_while_locked"]
                      and iso["hard_links_unchanged_at_end"])
        ok = st == 0 and p["failed"] == 0 and not p["targets_without_result"] and p["targets"] > 0 and iso_ok
        res["tests"] = {"definition": RUN_DEF["tests"], "exit_status": st, "outcome": outcome(ok, True),
                        "isolation": iso, **p}
    else:
        res["tests"] = {"definition": RUN_DEF["tests"], "outcome": "not run"}
    for step in ("spec_check", "spec_check_three_parsers"):
        s = run_file(runs, step + ".log")
        if s is not None:
            p = parse_spec_check(s)
            st = run_status(runs, step)
            ok = st == 0 and p["all_ok"] and p["grammar_conflicts_total"] == 0 and len(p["lalr_tables"]) > 0
            res[step] = {"definition": RUN_DEF[step], "exit_status": st, "outcome": outcome(ok, True), **p}
        else:
            res[step] = {"definition": RUN_DEF[step], "outcome": "not run"}
    w = run_file(runs, "wasm.log")
    if w is not None:
        p = parse_wasm(w)
        st = run_status(runs, "wasm")
        ok = (st == 0 and p["fail_lines"] == 0 and p["total"] is not None and p["total"][0] == p["total"][1]
              and p["grammar"] is not None and p["grammar"][1] + p["checker"][1] == p["total"][1])
        res["wasm"] = {"definition": RUN_DEF["wasm"], "exit_status": st, "outcome": outcome(ok, True),
                       "builds": wasm_builds(), **p}
    else:
        res["wasm"] = {"definition": RUN_DEF["wasm"], "outcome": "not run"}
    local, ci = run_file(runs, "lean.log"), run_file(runs, "lean-audit.ci.log")
    if local is not None or ci is not None:
        is_ci = local is None
        p = parse_lean(ci if is_ci else local, is_ci)
        lc = lean_counts()
        r = {"definition": RUN_DEF["lean"], "source": "public CI job log" if is_ci else "local run of scripts/lean-audit.sh"}
        if is_ci:
            meta = json.loads(run_file(runs, "lean-audit.ci.json") or "{}")
            r.update({k: meta.get(k) for k in ("repo", "workflow", "job_name", "run_id", "job_id")})
            c = p.get("commit")
            tree_ci = git("rev-parse", f"{c}:spec/lean") if c else None
            rc = (run_file(runs, "run_commit") or "").strip() or "HEAD"
            tree_here = git("rev-parse", f"{rc}:spec/lean")
            r["commit"] = c
            r["spec_lean_tree"] = tree_ci
            r["spec_lean_tree_equal_to_run_commit"] = (tree_ci == tree_here) if tree_ci and tree_here else None
            ok_tree = r["spec_lean_tree_equal_to_run_commit"] is True
        else:
            r["exit_status"] = run_status(runs, "lean")
            r["commit"] = (run_file(runs, "run_commit") or "").strip() or git("rev-parse", "HEAD")
            ok_tree = r["exit_status"] == 0
        r.update(p)
        ok = (p["standard_axioms_only"] and p["build_completed"] and p["sorry_messages"] == 0
              and lc["lean_sorry_in_source"] == 0 and lc["lean_axiom_declarations"] == 0 and ok_tree)
        r["outcome"] = outcome(ok, True)
        res["lean"] = r
    else:
        res["lean"] = {"definition": RUN_DEF["lean"], "outcome": "not run"}
    return res


def canonical(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def summary(runs, base):
    head = git("rev-parse", "HEAD")
    rc = (run_file(runs, "run_commit") or "").strip() or head
    st = run_file(runs, "worktree_status")
    trees = {p: git("rev-parse", f"{rc}:{p}") for p in TESTED_TREES}
    s = {
        "schema": SCHEMA,
        "repo": REPO,
        "run_commit": rc,
        "run_commit_definition": "the commit checked out when scripts/verify-run.sh ran the commands (DIR/run_commit)",
        "run_worktree_clean": (st.strip() == "") if st is not None else None,
        "collector_commit": head,
        "collector_trees_equal_to_run_commit": all(git("rev-parse", f"HEAD:{p}") == trees[p] for p in TESTED_TREES),
        "trees": trees,
        "toolchains": toolchains(),
        "counts": source_counts(),
        "runs": runs_summary(runs),
    }
    if base:
        bc = git("rev-parse", base + "^{commit}")
        eq = {p: (git("rev-parse", f"{bc}:{p}") == s["trees"][p]) if bc else None for p in TESTED_TREES}
        s["main_commit"] = bc
        s["trees_equal_to_main_commit"] = eq
        s["commit"] = bc if bc and all(eq.values()) else rc
    else:
        s["commit"] = rc
    s["commit_definition"] = ("the commit of main whose tested trees equal those of the run commit, "
                              "when --base names one; otherwise the run commit")
    lean = s["runs"]["lean"]
    s["lean_commit"] = lean.get("commit")
    s["checks_not_passed"] = sorted(k for k, v in s["runs"].items() if v["outcome"] != "passed")
    if not s["collector_trees_equal_to_run_commit"]:
        s["checks_not_passed"].append("collector_trees_equal_to_run_commit")
    if s["run_worktree_clean"] is False:
        s["checks_not_passed"].append("run_worktree_clean")
    return s


def write_facts(path, runs, base, site_facts):
    s = summary(runs, base)
    digest = hashlib.sha256(canonical(s)).hexdigest()
    facts = {
        "source": "scripts/claim-counts.py --write in openIE-dev/rce; summary.counts are read from the source, summary.runs from the logs of scripts/verify-run.sh",
        "repo": REPO,
        "commit": (s["commit"] or "")[:8],
        "lean_commit": (s["lean_commit"] or "")[:8] or None,
        "spec_document_draft": s["counts"]["spec_document_draft"]["value"],
        "summary_sha256": digest,
        "summary_sha256_definition": "sha256 of the canonical JSON of summary: keys sorted, separators ',' and ':', UTF-8",
        "summary": s,
    }
    if site_facts:
        old = json.loads(pathlib.Path(site_facts).read_text())
        carried = {k: old[k] for k in SITE_KEYS if k in old}
        facts.update(carried)
        facts["site_keys_carried"] = sorted(carried)
        facts["site_keys_carried_from"] = "the site's earlier src/data/rce-facts.json (the notebook is generated on the site, not in this repository)"
    pathlib.Path(path).write_text(json.dumps(facts, indent=1, ensure_ascii=False) + "\n")
    return facts


def opt(name):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        sys.exit(f"claim-counts: {name} needs a value")
    return None


def main():
    runs, write, facts_path = opt("--runs"), opt("--write"), opt("--facts")
    base, site = opt("--base"), opt("--site-facts")
    if write:
        if not runs:
            sys.exit("claim-counts: --write needs --runs DIR (the logs of scripts/verify-run.sh)")
        f = write_facts(write, runs, base, site)
        s = f["summary"]
        print(f"claim-counts: wrote {write}")
        print(f"claim-counts: commit {s['commit']}, run commit {s['run_commit']}, lean commit {s['lean_commit']}")
        for k, v in s["runs"].items():
            print(f"claim-counts: {k}: {v['outcome']}")
        print(f"claim-counts: summary sha256 {f['summary_sha256']}")
        sys.exit(1 if s["checks_not_passed"] else 0)
    if "--check" in sys.argv:
        spec_out = run_file(runs, "spec_check.log") if runs else None
        c = counts(spec_out=spec_out)
        errs = check(c)
        if facts_path:
            if not runs:
                sys.exit("claim-counts: --facts needs --runs DIR")
            have = json.loads(pathlib.Path(facts_path).read_text())
            fresh = summary(runs, base)
            if canonical(have.get("summary")) != canonical(fresh):
                diff = [k for k in sorted(set(fresh) | set(have.get("summary", {})))
                        if canonical(fresh.get(k)) != canonical(have.get("summary", {}).get(k))]
                errs.append(f"{facts_path}: summary differs from a recomputation in {diff}")
            if have.get("summary_sha256") != hashlib.sha256(canonical(fresh)).hexdigest():
                errs.append(f"{facts_path}: summary_sha256 differs from the recomputed digest")
        for e in errs:
            print("MISMATCH:", e)
        print("claim-counts: %s" % ("FAIL" if errs else "all quoted figures match the source"))
        sys.exit(1 if errs else 0)
    c = counts()
    c.update({k: v["value"] for k, v in source_counts().items()})
    print(json.dumps(c, indent=1))


main()
