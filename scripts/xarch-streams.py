#!/usr/bin/env python3
"""Determinism streams (RCE-JEPA H8): seeded event streams over the example
programs, traced in one call per program, one digest line per program.

For every spec example that carries `example` lines, the event instances of
those lines form a pool; STREAMS seeded streams of 1 to 12 events drawn from
the pool are written one per line and traced in one run with
`rce trace streams FILE PATH`, and the SHA-256 of the trace text
(paths removed) is printed next to the program's name. scripts/xarch-hashes.sh prints these
lines into the golden set, so the parity workflow compares the hash of
every stream's chain on every target. The streams are a pure function of
the seed and the examples.

    xarch-streams.py RCE_BIN [RUNNER WORDS...]
"""
import hashlib, os, re, subprocess, sys, tempfile
from pathlib import Path

SEED = 0x5EED_D3AD_0008
STREAMS = int(os.environ.get("XARCH_STREAMS", "700"))
ROOT = Path(__file__).resolve().parent.parent


class Rng:
    def __init__(self, s):
        self.s = s & ((1 << 64) - 1)

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return z ^ (z >> 31)


def split_events(text):
    """Top-level comma split of an example's event list (payloads may hold commas)."""
    out, depth, cur, in_str = [], 0, "", False
    i = 0
    while i < len(text):
        c = text[i]
        if in_str:
            cur += c
            if c == "\\" and i + 1 < len(text):
                cur += text[i + 1]
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            cur += c
        elif c in "([{":
            depth += 1
            cur += c
        elif c in ")]}":
            depth -= 1
            cur += c
        elif c == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += c
        i += 1
    if cur.strip():
        out.append(cur.strip())
    return out


def pool_of(src):
    pool = []
    for line in src.split("\n"):
        m = re.match(r"\s*example\s+(.*?)\s*->", line)
        if m:
            for e in split_events(m.group(1)):
                if e not in pool:
                    pool.append(e)
    return pool


def main():
    rce = sys.argv[1]
    runner = sys.argv[2:]
    files = sorted((ROOT / "spec" / "examples").glob("*.rce"))
    rng = Rng(SEED)
    total = 0
    with tempfile.TemporaryDirectory() as d:
        for f in files:
            src = f.read_text(encoding="utf-8")
            pool = pool_of(src)
            if not pool:
                continue
            lines = []
            for _ in range(STREAMS):
                n = 1 + rng.next() % 12
                lines.append(", ".join(pool[rng.next() % len(pool)] for _ in range(n)))
            streams = Path(d) / (f.stem + ".streams")
            streams.write_text("\n".join(lines) + "\n", encoding="utf-8")
            r = subprocess.run([*runner, rce, "trace", "streams", str(f), str(streams)], capture_output=True, text=True, encoding="utf-8")
            text = (r.stdout + r.stderr).replace(str(f), "").replace(str(f).replace("\\", "/"), "").replace(d, "").replace(d.replace("\\", "/"), "")
            text = text.replace("\r\n", "\n")
            traces = text.count('"source":"stream@')
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            print(f"{h} streams {f.name} ({traces} traces of {STREAMS} streams, status {r.returncode})")
            total += traces
    print(f"# streams total {total} (seed {SEED:#x})", file=sys.stderr)


if __name__ == "__main__":
    main()
