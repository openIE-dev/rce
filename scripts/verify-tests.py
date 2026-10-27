#!/usr/bin/env python3
"""Run the reference checker's tests, and one of its binaries, from the files
of one build, so that a build of another commit into a shared target
directory cannot replace them while they run.

  scripts/verify-tests.py tests DIR
      writes DIR/tests.log (in the format `cargo test` prints), DIR/tests.status
      (0 when every target records 0 failed and every binary is unchanged at
      the end) and DIR/tests.binaries.json
  scripts/verify-tests.py bin NAME [FEATURES] DIR/STEP.log
      builds the binary NAME (with the cargo features FEATURES, if given), runs
      it in spec/checker from a hard link, and writes its output to
      DIR/STEP.log, its exit status to DIR/STEP.status and its sha256 to
      DIR/STEP.binary.json

Isolation. Cargo holds the lock on the target directory
(<target>/release/.cargo-lock) while it builds, not while tests run, so a
build of another commit can replace a binary between the build and its use.
1. The build runs (`cargo test --release --no-run`, or `cargo build --release
   --bin NAME`); the moment it ends, every file it lists is hard-linked into
   DIR/bin (the same inode, no copy). A link whose modification time is later
   than the end of the build is a file another build wrote; then the build
   runs again (at most three times).
2. Every target runs from its hard link.
3. The 11 test targets whose source names CARGO_BIN_EXE_rce run
   <target>/release/rce, a path compiled into them. They run while this
   script holds the target directory's lock, so no build can run meanwhile;
   if <target>/release/rce is not the file of this build, the build's file is
   put at that path for the duration (a rename of a hard link) and the file
   that was there is put back before the lock is released.
4. Doc tests run through `cargo test --release --doc`.
Each target runs with its working directory spec/checker and TMPDIR set to
DIR/tmp, in the order cargo uses (library, binaries, integration tests, each
by name), after the line cargo prints for it ("     Running ..."). At the end
the sha256 of every hard link is compared with the one recorded at the start.
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
    log.flush()
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


def build_and_link(args, select, linkdir, log):
    linkdir.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        msgs = cargo_json(args, log)
        end = time.time_ns()
        if msgs is None:
            return None
        res = select(msgs)
        links, late = {}, []
        for f in res["files"]:
            dst = linkdir / os.path.basename(f)
            if dst.exists():
                dst.unlink()
            os.link(f, dst)
            links[f] = dst
            if os.stat(dst).st_mtime_ns > end:
                late.append(f)
        if not late:
            res["links"] = links
            res["attempts"] = attempt + 1
            return res
        log.write(f"verify-tests: {len(late)} file(s) written by another build after this build ended; building again\n")
        log.flush()
    return None


def run_commit_short():
    return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=root,
                          capture_output=True, text=True).stdout.strip()


def tests(out):
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
        return {"arts": arts, "rce": rce, "files": [a["exe"] for a in arts] + [rce]}

    res = build_and_link(["test", "--release", "--no-run"], select, out / "bin", log)
    if res is None:
        (out / "tests.status").write_text("101\n")
        return 101
    arts, rce, links = res["arts"], res["rce"], res["links"]

    def label(a):
        rel = os.path.relpath(a["src"], checker)
        return ("unittests " + rel) if a["kind"] in ("lib", "bin") else rel

    def uses_rce(a):
        try:
            return "CARGO_BIN_EXE_rce" in pathlib.Path(a["src"]).read_text(errors="ignore")
        except OSError:
            return False

    record = {label(a): {"file": os.path.basename(a["exe"]), "sha256": sha256(links[a["exe"]]),
                         "uses_rce_binary": uses_rce(a)} for a in arts}
    my_rce = links[rce]
    rce_sha = sha256(my_rce)
    rce_embeds = run_commit_short().encode() in my_rce.read_bytes()
    statuses = []

    def run(a):
        log.write(f"     Running {label(a)} ({a['exe']})\n")
        log.flush()
        r = subprocess.run([str(links[a["exe"]])], cwd=checker, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.flush()
        statuses.append(r.returncode)

    locked = [a for a in arts if uses_rce(a)]
    lockf = open(pathlib.Path(rce).parent / ".cargo-lock", "a+")
    t_wait = time.time()
    fcntl.flock(lockf, fcntl.LOCK_EX)
    t0 = time.time()
    placed = os.stat(rce).st_ino != os.stat(my_rce).st_ino
    if placed:
        keep = out / "bin" / "rce.found-at-target"
        if keep.exists():
            keep.unlink()
        os.link(rce, keep)
        tmp = pathlib.Path(rce).with_name(".rce.verify-tests")
        if tmp.exists():
            tmp.unlink()
        os.link(my_rce, tmp)
        os.replace(tmp, rce)
    for a in locked:
        run(a)
    rce_same = os.stat(rce).st_ino == os.stat(my_rce).st_ino and sha256(rce) == rce_sha
    if placed:
        tmp = pathlib.Path(rce).with_name(".rce.verify-tests")
        if tmp.exists():
            tmp.unlink()
        os.link(keep, tmp)
        os.replace(tmp, rce)
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
        "build_attempts": res["attempts"],
        "targets_run_under_lock": [label(a) for a in locked],
        "seconds_waiting_for_lock": round(t0 - t_wait, 1),
        "seconds_under_lock": round(t1 - t0, 1),
        "rce_binary": {"sha256": rce_sha, "embeds_run_commit": rce_embeds,
                       "placed_at_target_for_locked_targets": placed,
                       "unchanged_while_locked": rce_same},
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
        return {"exe": exe, "files": [exe]}

    linkdir = pathlib.Path(stem + ".bin")
    res = build_and_link(args, select, linkdir, log)
    if res is None:
        pathlib.Path(stem + ".status").write_text("101\n")
        return 101
    link = res["links"][res["exe"]]
    digest = sha256(link)
    log.write(f"     Running `{os.path.relpath(res['exe'], checker)}` (features: {features or 'default'})\n")
    log.flush()
    r = subprocess.run([str(link)], cwd=checker, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    pathlib.Path(stem + ".status").write_text(f"{r.returncode}\n")
    pathlib.Path(stem + ".binary.json").write_text(json.dumps(
        {"binary": name, "features": features or "default", "sha256": digest,
         "build_attempts": res["attempts"], "unchanged_at_end": sha256(link) == digest}, indent=1) + "\n")
    return r.returncode


def main():
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "tests":
        return tests(pathlib.Path(a[1]).resolve())
    if a and a[0] == "bin" and len(a) in (3, 4):
        return binary(a[1], a[2] if len(a) == 4 else None, pathlib.Path(a[-1]).resolve())
    sys.exit(__doc__)


sys.exit(main())
