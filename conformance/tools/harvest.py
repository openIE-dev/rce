#!/usr/bin/env python3
"""Seed the suite from the engine's own corpus: spec/examples (and its
ill_typed / inadmissible / reject folders), conc06*, the .rce files under
checker/tests, inline programs in checker/tests/*.rs, and the golden traces
of spec/traces. Every program is labelled by the reference adapter,
deduplicated by the SHA-256 of its text, and written as a test file.
Golden trace heads are compared with the reference; a mismatch is a finding,
not a test.
  harvest.py SPEC_DIR OUT_DIR
"""
import hashlib, json, os, pathlib, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
sys.path.insert(0, os.path.dirname(__file__))
import rcetest
from refclient import Ref
from classify import classify

spec, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
ref = Ref()
seen, findings, n = set(), [], {"files": 0}

def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]

def put(area, name, src, meta):
    d = out / area / "harvested"; d.mkdir(parents=True, exist_ok=True)
    meta = {"id": f"{area}/harvested/{name}", "area": area, **meta}
    p = d / f"{name}.rce"
    k = 2
    while p.exists():
        p = d / f"{name}-{k}.rce"; meta["id"] = f"{area}/harvested/{name}-{k}"; k += 1
    rcetest.write(p, src, meta); n["files"] += 1

def example_lines(src):
    for i, l in enumerate(src.splitlines(), 1):
        m = re.match(r"\s*example\s+(.+?)\s*->", l)
        if m: yield i, m.group(1)

def label(src, origin, fixtures=None):
    h = hashlib.sha256(src.encode()).hexdigest()
    if h in seen: return
    seen.add(h)
    stem = slug(pathlib.Path(origin).stem)
    files = {"main.rce": src, **(fixtures or {})}
    r = ref.check(src, files)
    prov = f"harvested {origin}"
    if r.get("verdict") == "change":
        code = (r["diagnostics"][0].get("code") if r.get("diagnostics") else None)
        area = classify(code, origin, src)
        put(area, f"{stem}-change", src, {"clauses": [code] if code else [], "expect": "change",
            "diagnostic": code or "any", "provenance": prov})
        return
    if r.get("verdict") != "accept": return
    area = classify(None, origin, src)
    put(area, f"{stem}-accept", src, {"clauses": [], "expect": "accept", "provenance": prov})
    if fixtures: return
    if re.search(r"^fn main\(\)", src, re.M):
        rr = ref.run(src)
        if "error" not in rr:
            put(area, f"{stem}-main", src, {"clauses": [], "expect": "run", "outcome": rr["outcome"],
                "outputs": rr["outputs"], "provenance": prov})
    for line, evs in example_lines(src):
        rr = ref.run(src, evs)
        if "error" in rr: continue
        put("forms-log" if area == "evaluation-fuel" else area, f"{stem}-ex{line}", src,
            {"clauses": [], "expect": "run", "events": evs, "outcome": rr["outcome"], "outputs": rr["outputs"], "provenance": prov + f" example@{line}"})
        tr = ref.trace(src, evs)
        if "trace" in tr:
            put("forms-log", f"{stem}-ex{line}-trace", src, {"clauses": [], "expect": "trace", "events": evs,
                "trace_head": tr["trace"]["head"], "trace_lines": tr["trace"]["lines"], "provenance": prov + f" example@{line}"})

# 1. .rce files
roots = [spec / "examples", spec / "conc06", spec / "conc06b", spec / "checker" / "tests"]
for root in roots:
    for p in sorted(root.rglob("*.rce")):
        if "26_use_module" in str(p) or "/brick/" in str(p): continue
        label(p.read_text(encoding="utf-8", errors="replace"), str(p.relative_to(spec)))
# multi-module package
pkg = spec / "examples" / "26_use_module"
if pkg.exists():
    mods = sorted(pkg.rglob("*.rce"))
    main = [m for m in mods if "shop" not in str(m)]
    if main:
        fx = {}
        dst = out / "types-checking" / "harvested"; dst.mkdir(parents=True, exist_ok=True)
        for m in mods:
            if m in main: continue
            name = "use-module-" + slug(str(m.relative_to(pkg))) + "_FIXTURE.rce"
            (dst / name).write_text(m.read_text()); fx[name] = m.read_text()
        src = main[0].read_text()
        r = ref.check(src, {"main.rce": src, **fx})
        put("types-checking", "use-module-package", src, {"clauses": ["W-MOD-1"] if False else [], "expect": r.get("verdict"),
            "fixtures": list(fx), "provenance": "harvested examples/26_use_module"})

# 2. inline programs in checker/tests/*.rs (raw string literals)
RAW = re.compile(r'r(#+)"(.*?)"\1', re.S)
for p in sorted((spec / "checker" / "tests").glob("*.rs")):
    t = p.read_text(encoding="utf-8", errors="replace")
    for k, m in enumerate(RAW.finditer(t)):
        s = m.group(2)
        if not re.search(r"^(fn|state|event|step|struct|kind|intent|# rce-version|type|actor|trait|enum|module|use|const|error|lvar)\b", s.lstrip("\n"), re.M):
            continue
        if len(s) > 20000 or "{}" in s: continue
        s = s.lstrip("\n")
        label(s if s.endswith("\n") else s + "\n", f"checker/tests/{p.name}#{k}")

# 3. golden traces
for g in sorted((spec / "traces").glob("*.json")):
    d = json.loads(g.read_text())
    srcp = spec / "examples" / d["file"]
    if not srcp.exists(): continue
    src = srcp.read_text()
    for t in d["traces"]:
        evs = d.get("events")
        if t["source"].startswith("example@"):
            line = int(t["source"].split("@")[1])
            evs = dict(example_lines(src)).get(line)
        if not evs or not t["steps"]: continue
        golden = t["steps"][-1]["hash"]
        tr = ref.trace(src, evs)
        got = (tr.get("trace") or {}).get("head")
        if got != golden:
            findings.append({"kind": "golden-mismatch", "file": str(g.relative_to(spec)), "events": evs, "golden": golden, "reference": got})
            continue
        put("forms-log", f"golden-{slug(g.stem)}-{slug(t['source'])}", src, {"clauses": [], "expect": "trace", "events": evs,
            "trace_head": golden, "trace_lines": tr["trace"]["lines"], "provenance": f"golden spec/{g.relative_to(spec)}"})

json.dump(findings, open(out.parent / "findings" / "harvest-findings.json", "w"), indent=1)
print(json.dumps({"written": n["files"], "unique_programs": len(seen), "golden_mismatches": len(findings)}))
