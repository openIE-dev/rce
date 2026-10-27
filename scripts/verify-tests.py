#!/usr/bin/env python3
"""Run the reference checker's tests, and one of its binaries, from the files
of one build, so that a build of another commit into a shared target
directory cannot replace them while they run.

  scripts/verify-tests.py tests DIR
      writes DIR/tests.log (in the format `cargo test` prints), DIR/tests.status
      (0 when every target records 0 failed and every binary is unchanged at
      the end) and DIR/tests.binaries.json
  scripts/verify-tests.py bin NAME [FEATURES] DIR/STEP.log
      builds the binary NAME (with the cargo features FEATURES, if given),
      runs it in spec/checker, and writes its output to DIR/STEP.log, its exit
      status to DIR/STEP.status and its sha256 to DIR/STEP.binary.json

tests:
1. `cargo test --release --no-run --message-format=json` in spec/checker
   builds and lists every test executable and the `rce` binary.
2. Under the lock that cargo takes on the target directory
   (<target>/release/.cargo-lock), each listed file is checked to be the file
   the build wrote (inode and modification time; otherwise the build runs
   again), hard-linked into DIR/bin (the same inode, no copy), and its sha256
   recorded.
3. The targets whose source names CARGO_BIN_EXE_rce run <target>/release/rce;
   they run while the lock is still held, so no build can replace that
   binary. The lock is then released, and every other target runs from its
   hard link.
4. Doc tests run through `cargo test --release --doc`.
Each target runs with its working directory spec/checker and TMPDIR set to
DIR/tmp, in the order cargo uses (library, binaries, integration tests, each
by name), after the line cargo prints for it ("     Running ...").

bin: the binary runs while the lock is held, after the same check that it is
the file the build wrote.
"""
import fcntl, hashlib, json, os, pathlib, subprocess, sys, time

root = pathlib.Path(__file__).resolve().parent.parent
checker = root / "spec/checker"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cargo_json(args, log):
    r = subprocess.run(["cargo", *args, "--message-format=json-render-diagnostics"],
                       cwd=checker, capture_output=True, text=True)
    log.write(r.stderr)
    if r.returncode != 0:
        return None
    msgs = []
    for line in r.stdout.splitlines():
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("reason") == "compiler-artifact" and m.get("executable"):
            msgs.append(m)
    return msgs


def ident(p):
    s = os.stat(p)
    return (s.st_ino, s.st_mtime_ns, s.st_size)


def locked_build(args, select, log):
    """Build, then take the target directory's lock and return (lockfile,
    result) once every selected file is still the one the build wrote."""
    for _ in range(3):
        msgs = cargo_json(args, log)
        if msgs is None:
            return None, None
        res = select(msgs)
        files = res["files"]
        before = {f: ident(f) for f in files}
        lockf = open(pathlib.Path(res["lock_dir"]) / ".cargo-lock", "a+")
        fcntl.flock(lockf, fcntl.LOCK_EX)
        if all(ident(f) == before[f] for f in files):
            return lockf, res
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()
        log.write("verify-tests: a file changed between the build and the lock; building again\n")
    return None, None


def run_commit_short():
    return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=root,
                          capture_output=True, text=True).stdout.strip()


def tests(out):
    (out / "bin").mkdir(parents=True, exist_ok=True)
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(out / "tmp"), CARGO_MANIFEST_DIR=str(checker),
               CARGO_PKG_NAME="rce-spec-check")
    log = open(out / "tests.log", "w")
    order = {"lib": 0, "bin": 1, "test": 2}

    def select(msgs):
        arts, rce = [], None
        for m in msgs:
            t = m["target"]
            if m["profile"].get("test"):
                arts.append({"name": t["name"], "kind": t["kind"][0], "src": t["src_path"], "exe": m["executable"]})
            elif t["kind"] == ["bin"] and t["name"] == "rce":
                rce = m["executable"]
        arts.sort(key=lambda a: (order.get(a["kind"], 3), a["name"]))
        return {"arts": arts, "rce": rce, "files": [a["exe"] for a in arts] + [rce],
                "lock_dir": str(pathlib.Path(rce).parent)}

    lockf, res = locked_build(["test", "--release", "--no-run"], select, log)
    if lockf is None:
        (out / "tests.status").write_text("101\n")
        return 101
    arts, rce = res["arts"], res["rce"]

    def label(a):
        rel = os.path.relpath(a["src"], checker)
        return ("unittests " + rel) if a["kind"] in ("lib", "bin") else rel

    def uses_rce(a):
        try:
            return "CARGO_BIN_EXE_rce" in pathlib.Path(a["src"]).read_text(errors="ignore")
        except OSError:
            return False

    links, record = {}, {}
    for a in arts:
        dst = out / "bin" / os.path.basename(a["exe"])
        if dst.exists():
            dst.unlink()
        os.link(a["exe"], dst)
        links[a["exe"]] = dst
        record[label(a)] = {"file": os.path.basename(a["exe"]), "sha256": sha256(dst), "uses_rce_binary": uses_rce(a)}
    rce_sha = sha256(rce)
    rce_embeds = run_commit_short().encode() in pathlib.Path(rce).read_bytes()
    statuses = []

    def run(a):
        log.write(f"     Running {label(a)} ({a['exe']})\n")
        log.flush()
        r = subprocess.run([str(links[a["exe"]])], cwd=checker, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.flush()
        statuses.append(r.returncode)

    t0 = time.time()
    locked = [a for a in arts if uses_rce(a)]
    for a in locked:
        run(a)
    rce_same = sha256(rce) == rce_sha
    fcntl.flock(lockf, fcntl.LOCK_UN)
    lockf.close()
    t1 = time.time()
    for a in arts:
        if a not in locked:
            run(a)
    r = subprocess.run(["cargo", "test", "--release", "--doc"], cwd=checker, env=env, stdout=log, stderr=subprocess.STDOUT)
    statuses.append(r.returncode)
    unchanged = all(sha256(links[a["exe"]]) == record[label(a)]["sha256"] for a in arts)
    info = {
        "executables": len(arts),
        "targets_run_under_lock": [label(a) for a in locked],
        "seconds_under_lock": round(t1 - t0, 1),
        "rce_binary": {"sha256": rce_sha, "embeds_run_commit": rce_embeds, "unchanged_while_locked": rce_same},
        "hard_links_unchanged_at_end": unchanged,
        "per_target": record,
    }
    (out / "tests.binaries.json").write_text(json.dumps(info, indent=1) + "\n")
    ok = all(s == 0 for s in statuses) and unchanged and rce_embeds and rce_same
    st = 0 if ok else 101
    (out / "tests.status").write_text(f"{st}\n")
    log.close()
    return st


def binary(name, features, logpath):
    stem = str(logpath)[:-4] if str(logpath).endswith(".log") else str(logpath)
    log = open(logpath, "w")
    args = ["build", "--release", "--bin", name] + (["--features", features] if features else [])

    def select(msgs):
        exe = next(m["executable"] for m in msgs if m["target"]["name"] == name and m["target"]["kind"] == ["bin"])
        return {"exe": exe, "files": [exe], "lock_dir": str(pathlib.Path(exe).parent)}

    lockf, res = locked_build(args, select, log)
    if lockf is None:
        pathlib.Path(stem + ".status").write_text("101\n")
        return 101
    log.write(f"     Running `{os.path.relpath(res['exe'], checker)}` (features: {features or 'default'})\n")
    log.flush()
    digest = sha256(res["exe"])
    r = subprocess.run([res["exe"]], cwd=checker, stdout=log, stderr=subprocess.STDOUT)
    fcntl.flock(lockf, fcntl.LOCK_UN)
    lockf.close()
    log.close()
    pathlib.Path(stem + ".status").write_text(f"{r.returncode}\n")
    pathlib.Path(stem + ".binary.json").write_text(json.dumps(
        {"binary": name, "features": features or "default", "sha256": digest}, indent=1) + "\n")
    return r.returncode


def main():
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "tests":
        return tests(pathlib.Path(a[1]).resolve())
    if a and a[0] == "bin" and len(a) in (3, 4):
        return binary(a[1], a[2] if len(a) == 4 else None, pathlib.Path(a[-1]).resolve())
    sys.exit(__doc__)


sys.exit(main())
