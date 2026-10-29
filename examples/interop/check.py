#!/usr/bin/env python3
"""The interop test: every embedding against `rce run`, byte for byte.

For each program and stream of streams.json, the expected bytes are what
`rce run FILE EVENTS` prints (the reference). Each embedding emits the
program with `rce emit LANG FILE`, drives the emitted step function from its
host library's event source, and prints the transcript; a transcript passes
only when it equals the reference exactly. The hand-written idiomatic
versions in hand/ are run through the same comparison.

  python3 examples/interop/check.py [LANG...]     LANG: python, fastapi, numpy,
                                                  rust, js, react, browser,
                                                  hand-python, hand-js, hand-rust
Env: RCE (the rce binary; default `rce`), INTEROP_OUT (work directory;
default examples/interop/out). A language whose toolchain or library is
absent is reported as `not run`, never as a pass; INTEROP_REQUIRE=LANG,..
makes that a failure.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RCE = os.environ.get("RCE", "rce")
OUT = os.path.abspath(os.environ.get("INTEROP_OUT", os.path.join(HERE, "out")))
REQUIRE = [x for x in os.environ.get("INTEROP_REQUIRE", "").split(",") if x]


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def rce_value(x):
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, str):
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, float):
        return repr(x)
    return str(x)


def rce_events(stream):
    """The stream in the event syntax of `rce run` (12.1)."""
    out = []
    for ev in stream:
        fs = [f"{k}: {rce_value(v)}" for k, v in ev.items() if k != "type"]
        out.append(ev["type"] + (f"({', '.join(fs)})" if fs else ""))
    return ", ".join(out)


def reference(prog, stream):
    r = sh([RCE, "run", os.path.join(ROOT, prog["file"]), rce_events(stream)])
    if not r.stdout:
        raise SystemExit(f"rce run {prog['file']}: {r.stderr}")
    return r.stdout


def emit(lang, prog, ext):
    path = os.path.join(OUT, lang, f"{prog['name']}.{ext}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = sh([RCE, "emit", lang, os.path.join(ROOT, prog["file"])])
    if r.returncode != 0:
        raise SystemExit(f"rce emit {lang} {prog['file']}: {r.stderr}")
    with open(path, "w") as w:
        w.write(r.stdout)
    return path


def stream_file(prog, i, stream):
    path = os.path.join(OUT, "streams", f"{prog['name']}.{i}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as w:
        json.dump(stream, w, ensure_ascii=False)
    return path


def have(*cmd):
    try:
        return sh(list(cmd)).returncode == 0
    except OSError:
        return False


def cargo(crate, target, exe):
    """Build a crate (offline when the registry cache has its crates) and return its executable."""
    target = os.path.join(OUT, target)
    for extra in (["--offline"], []):
        r = sh(["cargo", "build", "--release", "--quiet", "--locked", "--target-dir", target] + extra, cwd=crate)
        if r.returncode == 0:
            return os.path.join(target, "release", exe)
    raise SystemExit(f"cargo build {crate}: " + r.stderr[-3000:])


# ---- one runner per embedding: prepare() once, then run(prog, i, stream_path) -> transcript
class Python:
    name = "python"
    def ok(self):
        return have(sys.executable, "--version")
    def prepare(self, progs):
        self.mods = {p["name"]: emit("python", p, "py") for p in progs}
    def run(self, prog, i, sp):
        return sh([sys.executable, os.path.join(HERE, "python", "service.py"), self.mods[prog["name"]], sp])


class FastAPI(Python):
    name = "fastapi"
    def ok(self):
        return have(sys.executable, "-c", "import fastapi, fastapi.testclient")
    def run(self, prog, i, sp):
        return sh([sys.executable, os.path.join(HERE, "python", "fastapi_app.py"), self.mods[prog["name"]], sp])


class Numpy(Python):
    """numpy scalars handed to the adapter directly (no JSON)."""
    name = "numpy"
    def ok(self):
        return have(sys.executable, "-c", "import numpy")
    def run(self, prog, i, sp):
        return sh([sys.executable, os.path.join(HERE, "python", "numpy_drive.py"), self.mods[prog["name"]], sp])


class Js:
    name = "js"
    script = "service.mjs"
    def ok(self):
        return have("node", "--version")
    def prepare(self, progs):
        self.mods = {p["name"]: emit("js", p, "mjs") for p in progs}
    def run(self, prog, i, sp):
        return sh(["node", os.path.join(HERE, "js", self.script), self.mods[prog["name"]], sp])


class React(Js):
    name = "react"
    script = "react_drive.mjs"
    def ok(self):
        return have("node", "--version") and os.path.isdir(os.path.join(HERE, "js", "node_modules", "react-test-renderer"))


class Browser(Js):
    """The emitted module in a realm with no `process` and no Node modules."""
    name = "browser"
    def run(self, prog, i, sp):
        return sh(["node", "--experimental-vm-modules", "--no-warnings", os.path.join(HERE, "js", "browser_drive.mjs"), self.mods[prog["name"]], sp])


class Rust:
    name = "rust"
    def ok(self):
        return have("cargo", "--version")
    def prepare(self, progs):
        gen = os.path.join(HERE, "rust", "src", "gen")
        os.makedirs(gen, exist_ok=True)
        for p in progs:
            src = emit("rust", p, "rs")
            shutil.copy(src, os.path.join(gen, f"{p['name']}.rs"))
        self.exe = cargo(os.path.join(HERE, "rust"), "rust-target", "rce-interop")
    def run(self, prog, i, sp):
        return sh([self.exe, "drive", prog["name"], sp])


class HandPython:
    """The hand-written idiomatic versions (hand/), through the same comparison."""
    name = "hand-python"
    def ok(self):
        return True
    def prepare(self, progs):
        pass
    def run(self, prog, i, sp):
        return sh([sys.executable, os.path.join(HERE, "hand", "python", "drive.py"), prog["name"], sp])


class HandJs(HandPython):
    name = "hand-js"
    def ok(self):
        return have("node", "--version")
    def run(self, prog, i, sp):
        return sh(["node", os.path.join(HERE, "hand", "js", "drive.mjs"), prog["name"], sp])


class HandRust(HandPython):
    name = "hand-rust"
    def ok(self):
        return have("cargo", "--version")
    def prepare(self, progs):
        self.exe = cargo(os.path.join(HERE, "hand", "rust"), "hand-rust-target", "rce-interop-hand")
    def run(self, prog, i, sp):
        return sh([self.exe, "drive", prog["name"], sp])


RUNNERS = {r.name: r for r in (Python(), FastAPI(), Numpy(), Rust(), Js(), React(), Browser(), HandPython(), HandJs(), HandRust())}


def main():
    names = sys.argv[1:] or list(RUNNERS)
    progs = json.load(open(os.path.join(HERE, "streams.json")))["programs"]
    cases = []
    for p in progs:
        for i, s in enumerate(p["streams"]):
            cases.append((p, i, s, reference(p, s), stream_file(p, i, s)))
    fails, report = 0, []
    for n in names:
        r = RUNNERS[n]
        if not r.ok():
            report.append(f"{n}: not run (toolchain or library absent)")
            if n in REQUIRE:
                fails += 1
            continue
        r.prepare(progs)
        good = 0
        for p, i, s, want, sp in cases:
            got = r.run(p, i, sp)
            if got.returncode == 0 and got.stdout == want:
                good += 1
                continue
            fails += 1
            print(f"FAIL {n} {p['name']} stream {i}: exit {got.returncode}\n{got.stderr[-2000:]}--- want\n{want}--- got\n{got.stdout}")
        report.append(f"{n}: {good}/{len(cases)} transcripts equal to rce run")
    print("\n".join(report))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
