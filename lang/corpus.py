#!/usr/bin/env python3
"""Pinned corpora for the first lifts: admit programs and verify recorded
observations (spec/proposals/rce-form.md section 20: the oracle record
rce-oracle/1, the corpus manifest rce-corpus/1, the observation rce-obs/1
and the corpus rules K1 to K7).

    python3 lang/corpus.py pins LANG
        Compute the toolchain digests on this host, print them as the
        [toolchain] table of an oracle record, and compare them with the
        record. Exit 0 when every pin holds, 2 when one does not.

    python3 lang/corpus.py admit LANG DIR
        Run every program DIR/<tier>/<item>.<ext> under the oracle record of
        lang/LANG/oracle/, as many times as its `runs` says and once under
        each perturbation it lists. A program whose observations all agree
        is admitted: its bytes are stored under their SHA-256 in
        lang/LANG/corpus/files/, its observation in corpus/expected/, its
        case in corpus/manifest.toml, and a line in corpus/admission.txt.
        A program whose runs differ, or that a check below keeps out, is
        listed in admission.txt with its class or reason and not admitted.

    python3 lang/corpus.py verify LANG [ITEM...]
        Check each case of the manifest (or the named items) against rule
        K1, run it under the oracle `runs` times, and print MATCH when every
        run gives the recorded observation. A run that does not reproduce it
        (another observation, a run past its limit, a failed build) is
        ORACLE-FAIL with the first differing byte or field: the oracle, not
        a lift, is then broken (section 20). A case that cannot be run (a
        pin that does not hold, an unreadable file, an expected file that
        breaks K1) is NOT-RUN. Exit 0 when every case matches, 1 when one is
        ORACLE-FAIL and none is NOT-RUN, 2 when one is NOT-RUN. DIFFERS, the
        result of comparing a lift with the oracle, is not a result of this
        command.

Only Python's standard library is used (3.11 or later, for tomllib). The
output has no times and no host paths, so two runs on hosts that hold the
pins print the same bytes.
"""
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LANG_DIR = Path(__file__).resolve().parent
TIERS = ("basics", "algorithms", "boundaries", "generated")
LICENCE = "MIT OR Apache-2.0"
LICENCE_LINE = "SPDX-License-Identifier: " + LICENCE
SOURCE = "openIE-dev/rce, written for this corpus"
GENERATOR = LANG_DIR / "generate.py"
PROFILE = "command"
# The one input of every case so far: no arguments, no standard input and no
# environment beyond the oracle's pins.
INPUTS = [{"argv": [], "env": {}, "stdin_hex": ""}]
WORKERS = max(1, min(4, os.cpu_count() or 1))


# ---- canonical JSON (spec 6.1.1) and keys --------------------------------

def text(s):
    out = ['"']
    for c in s:
        o = ord(c)
        if c == '"':
            out.append('\\"')
        elif c == "\\":
            out.append("\\\\")
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        elif o < 0x20:
            out.append("\\u%04x" % o)
        else:
            out.append(c)
    out.append('"')
    return "".join(out)


def canon(v):
    """Canonical JSON of a record value: null, bool, int, text, list, record.
    These records hold no floats, so a float raises an error rather
    than being written."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        if not -(2 ** 63) <= v < 2 ** 64:
            raise ValueError("integer outside the range of 6.1.1")
        return str(v)
    if isinstance(v, str):
        return text(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(canon(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(text(k) + ":" + canon(v[k]) for k in sorted(v)) + "}"
    raise TypeError("no canonical JSON here for a value of type " + type(v).__name__)


def key(v):
    """The lower-case hexadecimal SHA-256 of a value's canonical JSON."""
    return hashlib.sha256(canon(v).encode("utf-8")).hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(root, skip):
    """SHA-256 over the lines "<digest>  <path>\\n" of every entry under root,
    sorted by path: a regular file's digest is the SHA-256 of its bytes, a
    symbolic link's is "link:" and its target. Directories named in skip are
    left out wherever they occur."""
    entries = []

    def walk(d, rel):
        with os.scandir(d) as it:
            for e in it:
                r = rel + e.name
                if e.is_symlink():
                    entries.append((r, "link:" + os.readlink(e.path)))
                elif e.is_dir():
                    if e.name not in skip:
                        walk(e.path, r + "/")
                elif e.is_file():
                    entries.append((r, sha256_file(e.path)))

    walk(root, "")
    entries.sort(key=lambda t: t[0].encode("utf-8", "surrogateescape"))
    h = hashlib.sha256()
    for r, d in entries:
        h.update(f"{d}  {r}\n".encode("utf-8", "surrogateescape"))
    return h.hexdigest()


def shared_libraries(exe):
    """The shared objects that ldd lists for exe, each by the SHA-256 of the
    file it resolves to; {} for a statically linked executable."""
    ldd = shutil.which("ldd")
    if ldd is None:
        return None
    r = subprocess.run([ldd, exe], capture_output=True, text=True)
    said = r.stdout + r.stderr
    if "not a dynamic executable" in said or "statically linked" in said:
        return {}
    libs = {}
    for line in r.stdout.splitlines():
        m = re.match(r"\s*\S+\s+=>\s+(/\S+)\s+\(", line) or re.match(r"\s*(/\S+)\s+\(", line)
        if m:
            libs[m.group(1)] = sha256_file(os.path.realpath(m.group(1)))
    return libs


# ---- the oracle -----------------------------------------------------------

class Failure(Exception):
    pass


class Result:
    def __init__(self, status=None, stdout=b"", stderr_class="", failure=None):
        self.status, self.stdout, self.stderr_class, self.failure = status, stdout, stderr_class, failure


def remove_tree(path):
    """Remove a directory that a toolchain wrote into, read-only parts too."""
    for root, dirs, files in os.walk(path):
        for n in dirs + files:
            try:
                os.chmod(os.path.join(root, n), 0o700, follow_symlinks=False)
            except (OSError, NotImplementedError):
                pass
    shutil.rmtree(path, ignore_errors=True)


def first_line(b, run_dir):
    t = b.decode("utf-8", "replace").replace(run_dir, "{run}").strip().splitlines()
    return t[0] if t else ""


class Oracle:
    def __init__(self, lang):
        d = LANG_DIR / lang
        records = sorted((d / "oracle").glob("*.toml"))
        if len(records) != 1:
            raise Failure(f"lang/{lang}/oracle/ holds {len(records)} records; this script runs one")
        self.lang, self.dir, self.path = lang, d, records[0]
        self.rec = tomllib.loads(self.path.read_text(encoding="utf-8"))
        if self.rec.get("format") != "rce-oracle/1" or self.rec.get("lang") != lang:
            raise Failure(f"{self.rel(self.path)} is not an rce-oracle/1 record for {lang}")
        self.key = key(self.rec)
        self.name = self.rec["name"]
        obs = tomllib.loads((d / "observe.toml").read_text(encoding="utf-8"))
        if obs.get("format") != "rce-observe/1" or obs.get("lang") != lang:
            raise Failure(f"lang/{lang}/observe.toml is not an rce-observe/1 record for {lang}")
        self.observe = obs
        self.rows = []
        for r in obs.get("class", []):
            after = re.compile(r["after"], re.M) if "after" in r else None
            self.rows.append((r["id"], after, re.compile(r["pattern"], re.M), r["class"]))
        self.unpinned = [(u["source"], re.compile(u["pattern"], re.M)) for u in self.rec.get("unpinned", [])]
        self.tmp = None
        self.net = None

    def rel(self, p):
        return str(Path(p).resolve().relative_to(LANG_DIR.parent))

    # -- pins

    def computed_toolchain(self):
        tc = self.rec["toolchain"]
        exe = tc["executable"]
        found = {"executable": exe}
        found["sha256"] = sha256_file(exe) if os.path.isfile(exe) else "missing"
        if "tree" in tc:
            found["tree"] = tc["tree"]
            found["tree_skip"] = tc.get("tree_skip", [])
            found["tree_sha256"] = tree_sha256(tc["tree"], set(found["tree_skip"])) if os.path.isdir(tc["tree"]) else "missing"
        if "absent" in tc:
            found["absent"] = [("exists: " + p) if os.path.lexists(p) else p for p in tc["absent"]]
        libs = shared_libraries(exe) if os.path.isfile(exe) else None
        if libs is not None and "extension_dirs" in tc:
            # The shared objects that the executable's extension modules load
            # when a program imports one: pinned with the executable's own.
            found["extension_dirs"] = tc["extension_dirs"]
            for sub in tc["extension_dirs"]:
                for ext in sorted(Path(tc.get("tree", "/"), sub).glob("*.so")):
                    libs.update(shared_libraries(str(ext)) or {})
            libs = dict(sorted(libs.items()))
        found["libraries"] = libs if libs is not None else "ldd missing"
        found["version_output"] = self.version_output()
        return found

    def version_output(self):
        """The output of the record's version command, run in a new, empty
        directory with the build environment (or the run environment when
        there is no build)."""
        d = tempfile.mkdtemp(prefix="rce-version-")
        try:
            env = self.env(self.rec["pins"].get("build_env", self.rec["pins"]["env"]), {"run": d, "cache": d})
            r = subprocess.run(self.rec["version"], env=env, cwd=d, capture_output=True, timeout=60)
            return (r.stdout + r.stderr).decode("utf-8", "replace").strip()
        except OSError as e:
            return "cannot run: " + str(e)
        finally:
            remove_tree(d)

    def pin_differences(self):
        tc = self.rec["toolchain"]
        diffs = []
        for k, v in self.computed_toolchain().items():
            if tc.get(k) != v:
                diffs.append(f"toolchain.{k}: recorded {tc.get(k)!r}, found {v!r}")
        if self.net_prefix() is None:
            diffs.append("pins.network: no new network namespace can be made here (unshare --net)")
        cpus = len(os.sched_getaffinity(0))
        for n in self.rec.get("perturb", {}).get("affinity", []):
            if n > cpus:
                diffs.append(f"perturb.affinity: the record runs on {n} processors, this host has {cpus}")
        if self.rec.get("perturb", {}).get("affinity") and shutil.which("taskset") is None:
            diffs.append("perturb.affinity: taskset is missing")
        return diffs

    def net_prefix(self):
        if self.net is None:
            u, t = shutil.which("unshare"), shutil.which("true")
            self.net = []
            if u and t:
                for p in ([u, "--net"], [u, "--map-root-user", "--net"]):
                    if subprocess.run(p + ["--", t], capture_output=True).returncode == 0:
                        self.net = p + ["--"]
                        break
        return self.net or None

    # -- one run

    def env(self, table, vars):
        return {k: self.subst_one(v, vars) for k, v in table.items()}

    @staticmethod
    def subst_one(a, vars):
        for k, v in vars.items():
            a = a.replace("{" + k + "}", v)
        if re.search(r"\{[a-z]+\}", a):
            raise Failure("unknown template variable in " + a)
        return a

    def subst(self, argv, vars):
        return [self.subst_one(a, vars) for a in argv]

    def perturbations(self):
        p = self.rec.get("perturb", {})
        out = []
        if p.get("env_order") == "reversed":
            out.append(("env_order", ("env_order", None)))
        for n in p.get("affinity", []):
            out.append((f"affinity:{n}", ("affinity", n)))
        for e in p.get("env", []):
            out.append(("env:" + ",".join(f"{k}={v}" for k, v in e.items()), ("env", e)))
        return out

    @staticmethod
    def execute(argv, env, cwd, stdin, limit):
        p = subprocess.Popen(argv, env=env, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
        try:
            out, err = p.communicate(stdin, timeout=limit)
            return p.returncode, out, err, False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = p.communicate()
            return p.returncode, out, err, True

    def run(self, src, perturb=None):
        d = os.path.realpath(tempfile.mkdtemp(prefix="run-", dir=self.tmp))
        try:
            name = self.rec["source_name"]
            Path(d, name).write_bytes(src)
            vars = {"src": name, "bin": "./main", "run": d, "cache": str(self.cache)}
            pins = self.rec["pins"]
            if self.rec.get("build"):
                code, _, err, late = self.execute(self.net + self.subst(self.rec["build"], vars),
                                                  self.env(pins["build_env"], vars), d, b"", self.rec["build_limit"])
                if late:
                    return Result(failure=f"the build ran past its limit of {self.rec['build_limit']} s")
                if code != 0:
                    return Result(failure=f"the build exited {code}: {first_line(err, d)}")
            env = self.env(pins["env"], vars)
            prefix = list(self.net)
            if perturb is not None:
                kind, val = perturb
                if kind == "env_order":
                    env = dict(reversed(list(env.items())))
                elif kind == "affinity":
                    cpus = sorted(os.sched_getaffinity(0))[:val]
                    prefix = [shutil.which("taskset"), "--cpu-list", ",".join(map(str, cpus))] + prefix
                elif kind == "env":
                    env = {**env, **val}
            code, out, err, late = self.execute(prefix + self.subst(self.rec["run"], vars), env, d,
                                                bytes.fromhex(INPUTS[0]["stdin_hex"]), self.rec["limit"])
            if late:
                return Result(failure=f"the run passed its limit of {self.rec['limit']} s")
            sig = -code if code < 0 else None
            status = 128 + sig if sig is not None else code
            return Result(status=status, stdout=out, stderr_class=self.classify(err, d, sig))
        finally:
            remove_tree(d)

    def classify(self, err, run_dir, sig):
        o = self.observe
        if sig is not None:
            return o["signal"].replace("{n}", str(sig))
        if not err:
            return o["empty"]
        t = err.decode("utf-8", "replace")
        for _id, after, pat, cls in self.rows:
            s = t
            if after is not None:
                ms = list(after.finditer(t))
                if not ms:
                    continue
                s = t[ms[-1].end():]
            m = pat.search(s)
            if m:
                return cls.replace("{1}", m.group(1) if m.re.groups else "")
        fixed = err.replace(run_dir.encode("utf-8"), b"{run}")
        return o["fallback"].replace("{sha256}", sha256_bytes(fixed))

    def __enter__(self):
        base = tempfile.mkdtemp(prefix="rce-corpus-")
        self.base = base
        self.tmp = os.path.join(base, "runs")
        self.cache = os.path.join(base, "cache")
        os.makedirs(self.tmp)
        os.makedirs(self.cache)
        self.net_prefix()
        return self

    def __exit__(self, *exc):
        remove_tree(self.base)


# ---- observations ---------------------------------------------------------

def case_key(sha):
    return key({"inputs": INPUTS, "profile": PROFILE, "sha256": sha})


def observation(ck, res):
    return {"case": ck, "d_head": None, "entered": [], "format": "rce-obs/1", "outcome": "done",
            "outputs": [], "profile": PROFILE, "status": res.status,
            "stderr_class": res.stderr_class, "stdout_hex": res.stdout.hex()}


OBS_FIELDS = set(observation("", Result()))


def first_difference(exp, got):
    a, b = bytes.fromhex(exp["stdout_hex"]), bytes.fromhex(got["stdout_hex"])
    if a != b:
        n = min(len(a), len(b))
        i = next((i for i in range(n) if a[i] != b[i]), n)
        ea = f"0x{a[i]:02x}" if i < len(a) else "the end of the output"
        fb = f"0x{b[i]:02x}" if i < len(b) else "the end of the output"
        return f"stdout byte {i}: expected {ea}, found {fb}"
    if exp["status"] != got["status"]:
        return f"exit status: expected {exp['status']}, found {got['status']}"
    if exp["stderr_class"] != got["stderr_class"]:
        return f"stderr class: expected {exp['stderr_class']!r}, found {got['stderr_class']!r}"
    return None


# ---- the manifest ---------------------------------------------------------

def q(s):
    return text(s)


def write_manifest(path, lang, oracle_rel, cases):
    order = {t: i for i, t in enumerate(TIERS)}
    cases = sorted(cases, key=lambda c: (order[c["tier"]], c["provenance"]["item"]))
    lines = [
        "# The pinned corpus of " + lang + ": one case per program, each stored under its",
        "# SHA-256 and run with no arguments, no standard input and the oracle's",
        "# environment. Written by lang/corpus.py admit; see lang/README.md.",
        'format  = "rce-corpus/1"',
        "lang    = " + q(lang),
        "oracles = [" + q(oracle_rel) + "]",
    ]
    for c in cases:
        p = c["provenance"]
        prov = ", ".join(f"{k} = {q(p[k])}" for k in ("source", "generator", "item", "licence") if k in p)
        lines += [
            "",
            "[[case]]",
            "file         = " + q(c["file"]),
            "sha256       = " + q(c["sha256"]),
            "tier         = " + q(c["tier"]),
            "profile      = " + q(c["profile"]),
            'inputs       = [{ argv = [], stdin_hex = "", env = {} }]',
            "expected     = " + q(c["expected"]),
            "expected_key = " + q(c["expected_key"]),
            "class        = " + q(c["class"]),
            "provenance   = { " + prov + " }",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_manifest(lang):
    p = LANG_DIR / lang / "corpus" / "manifest.toml"
    if not p.exists():
        return None
    m = tomllib.loads(p.read_text(encoding="utf-8"))
    if m.get("format") != "rce-corpus/1" or m.get("lang") != lang:
        raise Failure(f"lang/{lang}/corpus/manifest.toml is not an rce-corpus/1 manifest for {lang}")
    return m


# ---- admit ----------------------------------------------------------------

def examine(oracle, tier, item, src, known):
    sha = sha256_bytes(src)
    if sha in known:
        return {"verdict": "already", "sha": sha, "why": "already admitted as " + known[sha]}
    head = src.decode("utf-8", "replace").splitlines()[:3]
    if not any(LICENCE_LINE in line for line in head):
        return {"verdict": "out", "sha": sha, "class": "licence",
                "why": f"no line '{LICENCE_LINE}' among its first three lines (K2)"}
    body = src.decode("utf-8", "replace")
    for source, pat in oracle.unpinned:
        m = pat.search(body)
        if m:
            return {"verdict": "out", "sha": sha, "class": "unpinned:" + source,
                    "why": f"it names {m.group(0).strip()!r}, which reads the {source}, and this host does not pin the {source}"}
    ck = case_key(sha)
    obs = []
    for i in range(oracle.rec["runs"]):
        r = oracle.run(src)
        if r.failure:
            return {"verdict": "out", "sha": sha, "class": "oracle", "why": f"run {i + 1}: {r.failure}"}
        obs.append(observation(ck, r))
    for i in range(1, len(obs)):
        d = first_difference(obs[0], obs[i])
        if d:
            return {"verdict": "out", "sha": sha, "class": "oracle-nondet", "why": f"run {i + 1} differs from run 1 at {d}"}
    perturbed = []
    for name, p in oracle.perturbations():
        r = oracle.run(src, p)
        if r.failure:
            return {"verdict": "out", "sha": sha, "class": "oracle", "why": f"under {name}: {r.failure}"}
        o = observation(ck, r)
        d = first_difference(obs[0], o)
        if d:
            return {"verdict": "out", "sha": sha, "class": "nondet:" + name, "why": f"under {name} it differs at {d}"}
        perturbed.append([name, key(o)])
    cls = "resource" if obs[0]["stderr_class"].startswith("resource:") else "deterministic"
    return {"verdict": "in", "sha": sha, "class": cls, "obs": obs[0], "runs": [key(o) for o in obs],
            "perturbed": perturbed}


def admit(lang, src_dir):
    src_dir = Path(src_dir)
    oracle = Oracle(lang)
    diffs = oracle.pin_differences()
    if diffs:
        for d in diffs:
            print("NOT-RUN  " + d)
        return 2
    cdir = oracle.dir / "corpus"
    manifest = load_manifest(lang)
    cases = list(manifest["case"]) if manifest else []
    known = {c["sha256"]: c["provenance"]["item"] for c in cases}
    candidates = []
    seen = {}
    for entry in sorted(src_dir.iterdir()):
        if entry.name not in TIERS or not entry.is_dir():
            raise Failure(f"{entry} is not one of the tier directories {', '.join(TIERS)}")
    for tier in TIERS:
        t = src_dir / tier
        if not t.is_dir():
            continue
        for p in sorted(t.iterdir()):
            if not p.is_file() or not p.suffix:
                raise Failure(f"{p} is not a program file")
            src = p.read_bytes()
            sha = sha256_bytes(src)
            item = f"{tier}/{p.stem}"
            if sha in seen:
                raise Failure(f"{item} has the same bytes as {seen[sha]}")
            seen[sha] = item
            candidates.append((tier, item, p.suffix, src))
    if not candidates:
        raise Failure(f"no programs under {src_dir}")
    gen_sha = sha256_file(GENERATOR) if GENERATOR.exists() else None
    with oracle:
        if oracle.rec.get("build"):
            oracle.run(candidates[0][3])  # fills the build cache once before the parallel runs
        with ThreadPoolExecutor(WORKERS) as pool:
            results = list(pool.map(lambda c: examine(oracle, c[0], c[1], c[3], known), candidates))
    oracle_rel = "oracle/" + oracle.path.name
    log = []
    w = max(len(c[1]) for c in candidates)
    n_in = n_out = n_already = 0
    by_class = {}
    for (tier, item, ext, src), r in zip(candidates, results):
        if r["verdict"] == "already":
            n_already += 1
            log.append(f"already       {item:<{w}}  {r['sha']}  {r['why']}")
            continue
        if r["verdict"] == "out":
            n_out += 1
            log.append(f"not admitted  {item:<{w}}  {r['sha']}  {r['class']}: {r['why']}")
            continue
        n_in += 1
        by_class[r["class"]] = by_class.get(r["class"], 0) + 1
        sha = r["sha"]
        rel = f"files/{sha[:2]}/{sha}{ext}"
        (cdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (cdir / rel).write_bytes(src)
        exp = dict(r["obs"])
        exp["recorded"] = {"oracle": oracle.key, "runs": r["runs"], "perturbed": r["perturbed"]}
        erel = f"expected/{sha}.obs.json"
        (cdir / erel).parent.mkdir(parents=True, exist_ok=True)
        (cdir / erel).write_bytes(canon(exp).encode("utf-8"))
        prov = {"source": SOURCE, "item": item, "licence": LICENCE}
        if tier == "generated":
            if gen_sha is None:
                raise Failure("a generated program needs lang/generate.py")
            prov = {"source": "openIE-dev/rce, lang/generate.py", "generator": "sha256:" + gen_sha,
                    "item": item, "licence": LICENCE}
        cases.append({"file": rel, "sha256": sha, "tier": tier, "profile": PROFILE, "expected": erel,
                      "expected_key": key(r["obs"]), "class": r["class"], "provenance": prov})
        log.append(f"admitted      {item:<{w}}  {sha}  {r['class']}")
    write_manifest(cdir / "manifest.toml", lang, oracle_rel, cases)
    classes = ", ".join(f"{n} {c}" for c, n in sorted(by_class.items()))
    head = (f"admission of {len(candidates)} programs under {oracle.name} "
            f"(oracle record key {oracle.key})")
    tail = f"{n_in} admitted ({classes or 'none'}), {n_out} not admitted, {n_already} already admitted"
    block = [head] + log + [tail]
    adm = cdir / "admission.txt"
    prev = adm.read_text(encoding="utf-8") if adm.exists() else ""
    adm.write_text(prev + ("\n" if prev else "") + "\n".join(block) + "\n", encoding="utf-8")
    print("\n".join(block))
    return 0


# ---- verify ---------------------------------------------------------------

def verify_case(oracle, cdir, c):
    item = c["provenance"]["item"]
    if c.get("profile") != PROFILE or c.get("inputs") != INPUTS:
        return [("NOT-RUN", item, "this script runs only command cases with no arguments, input or environment of their own")]
    try:
        src = (cdir / c["file"]).read_bytes()
        exp_bytes = (cdir / c["expected"]).read_bytes()
        exp = json.loads(exp_bytes.decode("utf-8"))
        if not isinstance(exp, dict) or not isinstance(exp.get("recorded"), dict):
            raise ValueError("not an observation with a recorded part")
        rec = exp.pop("recorded")
        if set(exp) != OBS_FIELDS or exp["format"] != "rce-obs/1":
            raise ValueError("not an rce-obs/1 observation")
        bytes.fromhex(exp["stdout_hex"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [("NOT-RUN", item, f"its file or expected observation cannot be read: {type(e).__name__}")]
    if sha256_bytes(src) != c["sha256"]:
        return [("NOT-RUN", item, "the file's SHA-256 differs from the manifest (K3)")]
    if rec.get("oracle") != oracle.key:
        return [("NOT-RUN", item, f"recorded under oracle record key {rec.get('oracle')}, the record's key is {oracle.key}")]
    if canon({**exp, "recorded": rec}).encode("utf-8") != exp_bytes:
        return [("NOT-RUN", item, "the expected file is not canonical JSON")]
    if key(exp) != c["expected_key"] or exp.get("case") != case_key(c["sha256"]):
        return [("NOT-RUN", item, "the expected observation does not match the manifest's keys")]
    k1 = k1_difference(oracle, rec, c["expected_key"])
    if k1:
        return [("NOT-RUN", item, k1)]
    out = []
    for i in range(oracle.rec["runs"]):
        r = oracle.run(src)
        if r.failure:
            out.append(("ORACLE-FAIL", item, f"run {i + 1}: {r.failure}"))
            continue
        d = first_difference(exp, observation(exp["case"], r))
        out.append(("ORACLE-FAIL", item, f"run {i + 1}: {d}") if d else ("MATCH", item, f"run {i + 1}"))
    return out


def k1_difference(oracle, rec, expected_key):
    """Why an expected file's `recorded` breaks rule K1, or None. It must name
    the key of the observation of each of the record's runs and of the run
    under each of its perturbations, in the record's order; every one of
    them equals the expected observation's key, since a case is admitted
    only when all its runs agree."""
    runs = rec.get("runs")
    n = oracle.rec["runs"]
    if not isinstance(runs, list) or len(runs) != n:
        return f"the expected file does not name the observations of the record's {n} runs (K1)"
    if any(k != expected_key for k in runs):
        return "a run the expected file names has another key than the expected observation (K1)"
    names = [name for name, _ in oracle.perturbations()]
    perturbed = rec.get("perturbed")
    if (not isinstance(perturbed, list)
            or [p[0] if isinstance(p, list) and len(p) == 2 else None for p in perturbed] != names):
        return "the expected file does not name the observation under each perturbation of the record (K1)"
    if any(p[1] != expected_key for p in perturbed):
        return "a perturbed run the expected file names has another key than the expected observation (K1)"
    return None


def verify(lang, items):
    oracle = Oracle(lang)
    manifest = load_manifest(lang)
    if manifest is None:
        raise Failure(f"lang/{lang}/corpus/manifest.toml does not exist")
    cases = manifest["case"]
    if items:
        names = {c["provenance"]["item"] for c in cases}
        missing = [i for i in items if i not in names]
        if missing:
            raise Failure("not in the manifest: " + ", ".join(missing))
        cases = [c for c in cases if c["provenance"]["item"] in items]
    print(f"verify {lang}: {len(cases)} case{'' if len(cases) == 1 else 's'}, oracle {oracle.name} (record key {oracle.key}), "
          f"{oracle.rec['runs']} runs each")
    diffs = oracle.pin_differences()
    if diffs:
        for d in diffs:
            print("NOT-RUN  " + d)
        print(f"0 MATCH, 0 ORACLE-FAIL, {len(cases)} NOT-RUN")
        return 2
    cdir = oracle.dir / "corpus"
    with oracle:
        if oracle.rec.get("build") and cases:
            oracle.run((cdir / cases[0]["file"]).read_bytes())
        with ThreadPoolExecutor(WORKERS) as pool:
            results = list(pool.map(lambda c: verify_case(oracle, cdir, c), cases))
    counts = {"MATCH": 0, "ORACLE-FAIL": 0, "NOT-RUN": 0}
    width = max((len(c["provenance"]["item"]) for c in cases), default=0)
    for c, rs in zip(cases, results):
        verdicts = {v for v, _, _ in rs}
        if verdicts == {"MATCH"}:
            counts["MATCH"] += 1
            print(f"MATCH        {c['provenance']['item']:<{width}}  {c['expected_key']}")
            continue
        worst = "NOT-RUN" if "NOT-RUN" in verdicts else "ORACLE-FAIL"
        counts[worst] += 1
        for v, item, why in rs:
            if v != "MATCH":
                print(f"{v:<12} {item:<{width}}  {why}")
    print(", ".join(f"{n} {v}" for v, n in counts.items()))
    return 0 if counts["MATCH"] == len(cases) else (2 if counts["NOT-RUN"] else 1)


# ---- pins -----------------------------------------------------------------

def pins(lang):
    oracle = Oracle(lang)
    found = oracle.computed_toolchain()
    print("[toolchain]")
    tables = []
    for k, v in found.items():
        if isinstance(v, dict) and v:
            tables.append((k, v))
        elif isinstance(v, dict):
            print(f"{k} = {{}}")
        elif isinstance(v, list):
            print(f"{k} = [" + ", ".join(q(x) for x in v) + "]")
        else:
            print(f"{k} = {q(v)}")
    for k, v in tables:
        print(f"\n[toolchain.{k}]")
        for a, b in v.items():
            print(f"{q(a)} = {q(b)}")
    diffs = oracle.pin_differences()
    for d in diffs:
        print("# differs: " + d)
    print("# every pin holds" if not diffs else f"# {len(diffs)} pins do not hold")
    return 0 if not diffs else 2


def main(argv):
    try:
        if len(argv) == 2 and argv[0] == "pins":
            return pins(argv[1])
        if len(argv) == 3 and argv[0] == "admit":
            return admit(argv[1], argv[2])
        if len(argv) >= 2 and argv[0] == "verify":
            return verify(argv[1], argv[2:])
    except Failure as e:
        print("corpus.py: " + str(e), file=sys.stderr)
        return 2
    print(__doc__.strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
