#!/usr/bin/env python3
"""E1, determinism of lowering: build each declared program several times and compare the bytes.

For every configuration of matrix.json whose tool is on PATH, and every
program it names, the harness makes these builds, each in a fresh directory
holding only the copied source:

    A1, A2   twice in directory A                       -> same_dir
    B1       once in directory B (another absolute path) -> cross_dir (A1 against B1)
    Ar, Br   in A and in B with the documented path remapping -> remapped
    C1, C2   twice in A with SOURCE_DATE_EPOCH unset, at least 1.1 s apart -> same_dir_clock

Every other build has the declared environment: PATH and HOME from the
caller, LC_ALL=C, TZ=UTC, SOURCE_DATE_EPOCH=1735689600, and the
configuration's own variables. Every file the build leaves in its directory,
other than the source, is an output. For each pair of builds the harness
records, per output file, the two sizes and the number of differing bytes
(byte positions below the longer length at which the files differ or one
file has ended). It also counts how many times the absolute build directory
appears in the outputs of B1 and of Br, and keeps the SHA-256 of every output
of A1 and of Ar, so that results files from two machines can be compared.
For a differing ELF file it also counts the differing bytes per section.
For a differing Mach-O file (macOS) it counts the differing bytes per region
(sections, load commands, LC_UUID, and the pieces of __LINKEDIT: symbol table,
string table, code signature and the rest) and records each build's LC_UUID
and debug map (the N_SO and N_OSO stab entries: source and object paths, and
the object file's modification time).
A configuration with trace_env runs its remapped build once more in A and
in B with those variables (Go's GODEBUG=gocachehash=1 prints the inputs of
each action ID) and records the lines of that output which name the build
directory, the directory written as {dir}.

The results file is JSON and holds the machine, the toolchain versions and
one row per configuration and program. RESULTS.md is generated from it by
report.py; neither is written by hand.

    python3 e1.py run RESULTS_JSON [WORK_DIR]

WORK_DIR (default: a new directory under the system temporary directory)
holds the build directories and is deleted at the end. It is named by its
real path, with symbolic links resolved.
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "rce-compile-e1-results/2"
EPOCH = "1735689600"
TIMEOUT = 900


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def first_line(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return "unavailable: %s" % e.__class__.__name__
    out = (p.stdout or p.stderr).strip().splitlines()
    return out[0].strip() if out else "exit %d, no output" % p.returncode


def machine():
    cpu, model, osrel = "", "", ""
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        for line in open("/etc/os-release", encoding="utf-8"):
            if line.startswith("PRETTY_NAME="):
                osrel = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    if platform.system() == "Darwin":
        cpu = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        model = command_output(["sysctl", "-n", "hw.model"])
        name = command_output(["sw_vers", "-productName"])
        version = command_output(["sw_vers", "-productVersion"])
        build_v = command_output(["sw_vers", "-buildVersion"])
        if version:
            osrel = "%s %s%s" % (name or "macOS", version, " (%s)" % build_v if build_v else "")
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu": cpu,
        "model": model,
        "cpus": os.cpu_count(),
        "os": osrel,
        "python": platform.python_version(),
    }


def command_output(argv):
    """The command's standard output, stripped, or "" when it cannot run or fails."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return p.stdout.strip() if p.returncode == 0 else ""


def subst(argv, src, stem, d):
    return [a.replace("{src}", src).replace("{stem}", stem).replace("{dir}", d) for a in argv]


def snapshot(d, src):
    out = {}
    for root, _dirs, files in os.walk(d):
        for name in files:
            p = os.path.join(root, name)
            rel = os.path.relpath(p, d)
            if rel == os.path.normpath(src) or os.path.islink(p):
                continue
            with open(p, "rb") as f:
                out[rel] = f.read()
    return out


def build(cfg, argv, program, d, work, epoch=True, extra_env=None, want_stderr=False):
    """Build program in directory d with argv; return (outputs, error)."""
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    src = os.path.basename(program)
    stem = os.path.splitext(src)[0]
    sub = cfg.get("source_dir", "")
    os.makedirs(os.path.join(d, sub), exist_ok=True)
    shutil.copyfile(os.path.join(HERE, "programs", program), os.path.join(d, sub, src))
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/"), "LC_ALL": "C", "TZ": "UTC"}
    for k in ("RUSTUP_HOME", "CARGO_HOME"):
        if k in os.environ:
            env[k] = os.environ[k]
    if epoch:
        env["SOURCE_DATE_EPOCH"] = EPOCH
    env.update({k: subst([v], src, stem, d)[0] for k, v in cfg.get("env", {}).items()})
    env.update({k: subst([v], src, stem, d)[0] for k, v in (extra_env or {}).items()})
    fresh = []
    for k in cfg.get("fresh", []):
        f = tempfile.mkdtemp(prefix="fresh-", dir=work)
        env[k] = f
        fresh.append(f)
    try:
        p = subprocess.run(subst(argv, src, stem, d), cwd=d, env=env, capture_output=True, timeout=TIMEOUT)
        if p.returncode != 0:
            msg = (p.stderr or p.stdout).decode("utf-8", "replace").strip().splitlines()
            return None, "exit %d: %s" % (p.returncode, msg[-1] if msg else "")
        if want_stderr:
            return snapshot(d, os.path.join(sub, src)), p.stderr.decode("utf-8", "replace")
        return snapshot(d, os.path.join(sub, src)), None
    except subprocess.TimeoutExpired:
        return None, "timeout after %d s" % TIMEOUT
    finally:
        for f in fresh:
            shutil.rmtree(f, ignore_errors=True)


def differing_bytes(a, b):
    n = min(len(a), len(b))
    diff = abs(len(a) - len(b))
    if a[:n] == b[:n]:
        return diff
    # compare in blocks; count differing positions exactly
    i = 0
    step = 1 << 16
    while i < n:
        x, y = a[i:i + step], b[i:i + step]
        if x != y:
            diff += sum(1 for p, q in zip(x, y) if p != q)
        i += step
    return diff


def elf_sections(data):
    """(name, file offset, size) of each section of a 64-bit little-endian ELF file, else []."""
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return []
    shoff = int.from_bytes(data[0x28:0x30], "little")
    shentsize = int.from_bytes(data[0x3A:0x3C], "little")
    shnum = int.from_bytes(data[0x3C:0x3E], "little")
    shstrndx = int.from_bytes(data[0x3E:0x40], "little")
    if shoff == 0 or shoff + shnum * shentsize > len(data) or shstrndx >= shnum:
        return []
    hdrs = []
    for i in range(shnum):
        h = data[shoff + i * shentsize: shoff + (i + 1) * shentsize]
        hdrs.append((int.from_bytes(h[0:4], "little"), int.from_bytes(h[4:8], "little"),
                     int.from_bytes(h[0x18:0x20], "little"), int.from_bytes(h[0x20:0x28], "little")))
    stroff = hdrs[shstrndx][2]
    out = []
    for name, typ, off, size in hdrs:
        if typ == 8:  # SHT_NOBITS occupies no file bytes
            continue
        end = data.find(b"\0", stroff + name)
        out.append((data[stroff + name:end].decode("ascii", "replace"), off, size))
    return out


def diff_sections(a, b):
    """Count the differing byte positions of two equal-length ELF files per section of the first."""
    if len(a) != len(b):
        return None
    secs = elf_sections(a)
    if not secs:
        return None
    counts = {}
    for i, (p, q) in enumerate(zip(a, b)):
        if p != q:
            name = next((n for n, off, size in secs if off <= i < off + size), "(outside sections)")
            counts[name] = counts.get(name, 0) + 1
    return counts


# Mach-O: the file format of macOS executables, object files and the DWARF
# file inside a .dSYM bundle. Only thin 64-bit little-endian files (arm64,
# x86_64) are read; anything else gets no attribution.
MH_MAGIC_64 = 0xFEEDFACF
LC_NAMES = {
    0x2: "LC_SYMTAB", 0xB: "LC_DYSYMTAB", 0xC: "LC_LOAD_DYLIB", 0xE: "LC_LOAD_DYLINKER",
    0x19: "LC_SEGMENT_64", 0x1B: "LC_UUID", 0x1D: "LC_CODE_SIGNATURE", 0x1E: "LC_SEGMENT_SPLIT_INFO",
    0x26: "LC_FUNCTION_STARTS", 0x29: "LC_DATA_IN_CODE", 0x2A: "LC_SOURCE_VERSION",
    0x2B: "LC_DYLIB_CODE_SIGN_DRS", 0x2E: "LC_LINKER_OPTIMIZATION_HINT", 0x31: "LC_NOTE",
    0x32: "LC_BUILD_VERSION", 0x36: "LC_ATOM_INFO", 0x22: "LC_DYLD_INFO",
    0x80000022: "LC_DYLD_INFO_ONLY", 0x80000028: "LC_MAIN", 0x8000001C: "LC_RPATH",
    0x80000033: "LC_DYLD_EXPORTS_TRIE", 0x80000034: "LC_DYLD_CHAINED_FIXUPS",
}
# load commands that are a linkedit_data_command: (dataoff, datasize) at bytes 8..16
LINKEDIT_DATA = {
    0x1D: "code signature", 0x1E: "segment split info", 0x26: "function starts",
    0x29: "data in code", 0x2B: "dylib code signing DRs", 0x2E: "linker optimisation hints",
    0x36: "atom info", 0x80000033: "exports trie", 0x80000034: "chained fixups",
}
ZEROFILL = (0x1, 0xC, 0x12)  # S_ZEROFILL, S_GB_ZEROFILL, S_THREAD_LOCAL_ZEROFILL
N_SO, N_OSO = 0x64, 0x66


def _u32(d, o):
    return int.from_bytes(d[o:o + 4], "little")


def _u64(d, o):
    return int.from_bytes(d[o:o + 8], "little")


def _name16(d, o):
    return d[o:o + 16].split(b"\0", 1)[0].decode("ascii", "replace")


def macho_load_commands(data):
    """(cmd, offset, size) of each load command of a thin 64-bit little-endian Mach-O file, else None."""
    if len(data) < 32 or _u32(data, 0) != MH_MAGIC_64:
        return None
    ncmds, sizeofcmds = _u32(data, 16), _u32(data, 20)
    if 32 + sizeofcmds > len(data):
        return None
    out, off = [], 32
    for _ in range(ncmds):
        if off + 8 > 32 + sizeofcmds:
            return None
        cmd, size = _u32(data, off), _u32(data, off + 4)
        if size < 8 or off + size > 32 + sizeofcmds:
            return None
        out.append((cmd, off, size))
        off += size
    return out


def macho_regions(data):
    """Partition the bytes of a Mach-O file into named regions: [(name, start, end)], or None.

    The regions are the header; each load command (LC_UUID on its own, a
    segment command by its segment name); each section with file contents;
    the pieces of __LINKEDIT that the load commands locate (the symbol table,
    which holds the stabs of the debug map, the string table, the code
    signature and the others); and, for bytes in none of these, the segment
    that holds them ("segment __TEXT, other") or "outside segments". A byte
    belongs to one region only: where two pieces overlap, the one that starts
    first keeps the overlap.
    """
    cmds = macho_load_commands(data)
    if cmds is None:
        return None
    n = len(data)
    leaves, segments = [("header", 0, 32)], []

    def add(name, off, size):
        if size > 0 and off < n:
            leaves.append((name, off, min(n, off + size)))

    for cmd, off, size in cmds:
        lc = LC_NAMES.get(cmd, "LC_0x%X" % cmd)
        if cmd == 0x19:
            seg = _name16(data, off + 8)
            add("load command %s %s" % (lc, seg), off, size)
            fileoff, filesize = _u64(data, off + 40), _u64(data, off + 48)
            if filesize:
                segments.append((seg, fileoff, min(n, fileoff + filesize)))
            for i in range(_u32(data, off + 64)):
                h = off + 72 + 80 * i
                if h + 80 > off + size:
                    break
                sect, sseg = _name16(data, h), _name16(data, h + 16)
                ssize, soff, flags = _u64(data, h + 40), _u32(data, h + 48), _u32(data, h + 64)
                if filesize and soff and (flags & 0xFF) not in ZEROFILL:
                    add("%s,%s" % (sseg, sect), soff, ssize)
        else:
            add("LC_UUID" if cmd == 0x1B else "load command %s" % lc, off, size)
        if cmd == 0x2:
            symoff, nsyms, stroff, strsize = (_u32(data, off + 8 + 4 * k) for k in range(4))
            add("symbol table", symoff, 16 * nsyms)
            add("string table", stroff, strsize)
        elif cmd == 0xB:
            f = [_u32(data, off + 8 + 4 * k) for k in range(18)]
            add("dynamic symbol table: table of contents", f[6], 8 * f[7])
            add("dynamic symbol table: module table", f[8], 56 * f[9])
            add("dynamic symbol table: external references", f[10], 4 * f[11])
            add("indirect symbol table", f[12], 4 * f[13])
            add("dynamic symbol table: external relocations", f[14], 8 * f[15])
            add("dynamic symbol table: local relocations", f[16], 8 * f[17])
        elif cmd in LINKEDIT_DATA:
            add(LINKEDIT_DATA[cmd], _u32(data, off + 8), _u32(data, off + 12))
        elif cmd in (0x22, 0x80000022):
            for k, what in enumerate(("rebase", "bind", "weak bind", "lazy bind", "export")):
                add("dyld info: %s" % what, _u32(data, off + 8 + 8 * k), _u32(data, off + 12 + 8 * k))
    leaves.sort(key=lambda r: (r[1], r[2]))
    cut, cursor = [], 0
    for name, s, e in leaves:
        s = max(s, cursor)
        if s < e:
            cut.append((name, s, e))
            cursor = e
    out, cursor = [], 0

    def gap(s, e):
        while s < e:
            seg = next(((nm, a, b) for nm, a, b in segments if a <= s < b), None)
            if seg is None:
                nxt = min([a for _nm, a, _b in segments if a > s] + [e])
                out.append(("outside segments", s, nxt))
                s = nxt
            else:
                end = min(e, seg[2])
                out.append(("segment %s, other" % seg[0], s, end))
                s = end

    for name, s, e in cut:
        gap(cursor, s)
        out.append((name, s, e))
        cursor = e
    gap(cursor, n)
    return out


def diff_macho(a, b):
    """Differing bytes per region of two Mach-O files, or None when either is not one.

    Each region's bytes (all its pieces, in file order) are compared with
    the bytes of the same region in the other file, positionally from the
    region's own start, so a region that grows shifts no other region's
    count. A region missing in one file counts all its bytes.
    """
    ra, rb = macho_regions(a), macho_regions(b)
    if ra is None or rb is None:
        return None

    def gather(data, regions):
        g = {}
        for name, s, e in regions:
            g.setdefault(name, []).append(data[s:e])
        return {k: b"".join(v) for k, v in g.items()}

    ga, gb = gather(a, ra), gather(b, rb)
    counts = {}
    for name in set(ga) | set(gb):
        x, y = ga.get(name, b""), gb.get(name, b"")
        c = differing_bytes(x, y)
        if c:
            counts[name] = c
    return counts


def macho_debug_map(data, names=()):
    """LC_UUID and the N_SO and N_OSO stabs (the debug map) of a Mach-O file, or None.

    names is a list of (path, placeholder) pairs; each path is written as its
    placeholder in the recorded strings, in the order given.
    """
    cmds = macho_load_commands(data)
    if cmds is None:
        return None

    def named(s):
        for path, ph in names:
            s = s.replace(path, ph)
        return s

    uuid, so, oso = None, [], []
    for cmd, off, size in cmds:
        if cmd == 0x1B and size >= 24:
            uuid = data[off + 8:off + 24].hex()
        elif cmd == 0x2:
            symoff, nsyms, stroff, strsize = (_u32(data, off + 8 + 4 * k) for k in range(4))
            for i in range(nsyms):
                e = symoff + 16 * i
                if e + 16 > len(data):
                    break
                strx, ntype = _u32(data, e), data[e + 4]
                if ntype not in (N_SO, N_OSO) or strx >= strsize:
                    continue
                s0 = stroff + strx
                end = data.find(b"\0", s0, stroff + strsize)
                text = named(data[s0:end if end >= 0 else stroff + strsize].decode("utf-8", "replace"))
                if ntype == N_OSO:
                    oso.append([text, _u64(data, e + 8)])
                elif text:
                    so.append(text)
    return {"uuid": uuid, "so": so, "oso": oso}


def compare(x, y, names=()):
    """Compare two builds' outputs; names as for macho_debug_map, for the paths it records."""
    if x[1] or y[1]:
        return {"status": "build failed", "error": x[1] or y[1]}
    a, b = x[0], y[0]
    files = []
    for name in sorted(set(a) | set(b)):
        fa, fb = a.get(name), b.get(name)
        row = {
            "file": name,
            "size_a": None if fa is None else len(fa),
            "size_b": None if fb is None else len(fb),
        }
        if fa is None or fb is None:
            row["differing_bytes"] = len(fa if fa is not None else fb)
            row["missing_in"] = "a" if fa is None else "b"
        else:
            row["differing_bytes"] = differing_bytes(fa, fb)
            if row["differing_bytes"]:
                secs = diff_sections(fa, fb)
                if secs is not None:
                    row["sections"] = secs
                regions = diff_macho(fa, fb)
                if regions is not None:
                    row["macho_regions"] = regions
                    row["macho_debug_map"] = {"a": macho_debug_map(fa, names), "b": macho_debug_map(fb, names)}
        files.append(row)
    same = all(r["differing_bytes"] == 0 and "missing_in" not in r for r in files)
    return {"status": "identical" if same else "differs", "files": files}


def hashes(x):
    if x[0] is None:
        return None
    return {k: hashlib.sha256(v).hexdigest() for k, v in sorted(x[0].items())}


def path_copies(outputs, d):
    if outputs is None:
        return None
    needle = d.encode()
    return sum(v.count(needle) for v in outputs.values())


def run_row(cfg, program, work):
    a = os.path.join(work, "a", "build")
    b = os.path.join(work, "another-directory", "build")
    tmp = tempfile.gettempdir()
    names = [(a, "{dir}"), (b, "{dir}"), (os.path.realpath(tmp), "{tmpdir}"), (tmp, "{tmpdir}")]
    argv = cfg["build"]
    a1 = build(cfg, argv, program, a, work)
    a2 = build(cfg, argv, program, a, work)
    b1 = build(cfg, argv, program, b, work)
    row = {
        "config": cfg["id"],
        "tool": cfg["tool"],
        "language": cfg["language"],
        "program": program,
        "same_dir": compare(a1, a2, names),
        "cross_dir": compare(a1, b1, names),
        "path_copies_cross": path_copies(b1[0], b),
        "sha256": hashes(a1),
    }
    if cfg.get("remap"):
        ar = build(cfg, cfg["remap"], program, a, work, extra_env=cfg.get("remap_env"))
        br = build(cfg, cfg["remap"], program, b, work, extra_env=cfg.get("remap_env"))
        row["remapped"] = compare(ar, br, names)
        row["path_copies_remapped"] = path_copies(br[0], b)
        row["sha256_remapped"] = hashes(ar)
    else:
        row["remapped"] = {"status": "not tested", "error": cfg.get("remap_source", "")}
        row["path_copies_remapped"] = None
        row["sha256_remapped"] = None
    if cfg.get("trace_env"):
        argv_t = cfg["remap"] or argv
        lines = {}
        for name, d in (("a", a), ("b", b)):
            res = build(cfg, argv_t, program, d, work, extra_env=dict(cfg.get("remap_env", {}), **cfg["trace_env"]), want_stderr=True)
            if res[0] is None:
                lines[name] = None
                continue
            lines[name] = sorted({ln.replace(d, "{dir}") for ln in res[1].splitlines() if d in ln and not ln.startswith("HASH /")})
        row["trace"] = {"env": cfg["trace_env"], "lines_naming_dir": lines.get("a"),
                        "same_after_naming": lines.get("a") == lines.get("b")}
    t0 = time.monotonic()
    c1 = build(cfg, argv, program, a, work, epoch=False)
    wait = 1.1 - (time.monotonic() - t0)
    if wait > 0:
        time.sleep(wait)
    c2 = build(cfg, argv, program, a, work, epoch=False)
    row["same_dir_clock"] = compare(c1, c2, names)
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    return row


def main():
    args = sys.argv[1:]
    if len(args) not in (2, 3) or args[0] != "run":
        sys.exit("usage: python3 e1.py run RESULTS_JSON [WORK_DIR]")
    out = os.path.abspath(args[1])
    matrix = json.load(open(os.path.join(HERE, "matrix.json"), encoding="utf-8"))
    own_work = len(args) == 2
    work = tempfile.mkdtemp(prefix="e1-") if own_work else os.path.abspath(args[2])
    os.makedirs(work, exist_ok=True)
    # The build directory is named by its real path: a tool records the
    # directory that getcwd() returns, which has every symbolic link resolved
    # (macOS's temporary directory /var/folders is under the link /var ->
    # /private/var), so {dir}, the remappings and the path count use that path.
    work = os.path.realpath(work)
    toolchains, rows, absent = {}, [], []
    try:
        for cfg in matrix["configs"]:
            tool = cfg["tool"]
            if shutil.which(tool) is None:
                absent.append({"config": cfg["id"], "reason": "%s not on PATH" % tool})
                print("absent  %s (%s not on PATH)" % (cfg["id"], tool), flush=True)
                continue
            if "probe" in cfg and subprocess.run(cfg["probe"], capture_output=True).returncode != 0:
                absent.append({"config": cfg["id"], "reason": "probe failed: %s" % " ".join(cfg["probe"])})
                print("absent  %s (probe failed)" % cfg["id"], flush=True)
                continue
            toolchains.setdefault(tool, first_line(cfg["version"]))
            for program in cfg["programs"]:
                row = run_row(cfg, program, work)
                rows.append(row)
                print("%-34s %-22s same=%s cross=%s remap=%s clock=%s" % (
                    cfg["id"], program, row["same_dir"]["status"], row["cross_dir"]["status"],
                    row["remapped"]["status"], row["same_dir_clock"]["status"]), flush=True)
    finally:
        if own_work:
            shutil.rmtree(work, ignore_errors=True)
    result = {
        "schema": SCHEMA,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": machine(),
        "environment": {"LC_ALL": "C", "TZ": "UTC", "SOURCE_DATE_EPOCH": EPOCH},
        "harness": {
            "e1.py": sha256_file(os.path.abspath(__file__)),
            "matrix.json": sha256_file(os.path.join(HERE, "matrix.json")),
        },
        "toolchains": toolchains,
        "absent": absent,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True)
        f.write("\n")
    print("wrote %s: %d rows, %d configurations absent" % (out, len(rows), len(absent)))


if __name__ == "__main__":
    main()
