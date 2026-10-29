#!/usr/bin/env python3
"""Run the Python lift over the pinned corpus and compare each lifted
program's observation with the one recorded from CPython 3.13.16.

Usage (positional arguments only):

    verify_lift.py RCE [OUT]

RCE is the `rce` binary; OUT, when given, is a directory that receives
each lifted program (`<item>.rce`). One line per case of
lang/python3.13/corpus/manifest.toml, then the counts:

- MATCH: the lift applied, and the lifted program's observation (exit
  status, standard output byte for byte, standard error class) equals the
  recorded one.
- MATCH-RESOURCE: a case of class `resource` whose lifted program kept its
  event at a resource limit (section 20: such a case holds by definition).
- DIFFERS: the lift applied and the observations differ; the line names
  the first field that differs and, for standard output, the offset of
  the first differing byte.
- KEPT-RESOURCE: the lift applied and the lifted program kept its event at
  a limit (i64 overflow, a loop bound, the recursion bound) where CPython
  answered. Never counted as holding.
- REQUIRES-CHANGE: the lift reported constructs outside the subset; the
  line lists every rule it named.
- INTERNAL: the lifted text does not pass `rce check`, or `rce` failed: a
  defect of the lifter.

The command profile of the lifted program: `rce run FILE` runs `fn main()`;
every value it emits is text, and the bytes of standard output are the
UTF-8 of the emitted texts in order (the thin driver of section 19). The
exit status is 0 when main ends and the standard error is empty; a run that
stops has no exit status of the source's, so it is never equal.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CORPUS = os.path.join(ROOT, "lang", "python3.13", "corpus")
sys.path.insert(0, HERE)
import lift_py  # noqa: E402

LIMIT = 120   # seconds per run of a lifted program (the oracle's own limit is 10)
RESOURCE_TRAPS = ("overflow", "loop bound", "recursion bound", "fuel", "size", "budget")


def decode_rce_text(lit):
    """The text an RCE string literal as `rce run` shows it denotes
    (interp.rs show_text: \\0 \\t \\n \\r \\" \\\\ and \\u{h})."""
    assert lit[0] == '"' and lit[-1] == '"', lit
    s, i, o = lit[1:-1], 0, []
    while i < len(s):
        c = s[i]
        if c != "\\":
            o.append(c)
            i += 1
            continue
        n = s[i + 1]
        if n == "u":
            j = s.index("}", i)
            o.append(chr(int(s[i + 3:j], 16)))
            i = j + 1
            continue
        o.append({"0": "\0", "t": "\t", "n": "\n", "r": "\r", '"': '"', "\\": "\\"}[n])
        i += 2
    return "".join(o)


def observe(rce, path):
    """(status, stdout bytes, stderr class, stopped reason or None)."""
    try:
        p = subprocess.run([rce, "run", path], capture_output=True, timeout=LIMIT)
    except subprocess.TimeoutExpired:
        return None, None, None, f"rce run did not end within {LIMIT} s"
    out = p.stdout.decode("utf-8")
    if p.returncode not in (0, 1) or p.stderr:
        return None, None, None, f"rce exited {p.returncode}: {p.stderr.decode(errors='replace')[:200]}"
    data = []
    stopped = None
    for line in out.split("\n"):
        if not line:
            continue
        if line.startswith("emit \""):
            data.append(decode_rce_text(line[5:]).encode("utf-8"))
        elif line.startswith("stopped: "):
            stopped = line[9:]
        elif line.startswith("requires change"):
            return None, None, None, "rce: " + line
        else:
            return None, None, None, f"rce printed {line[:80]!r}"
    return 0, b"".join(data), "", stopped


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"offset {i}: oracle 0x{x:02x}, lift 0x{y:02x}"
    return f"offset {min(len(a), len(b))}: lengths {len(a)} and {len(b)}"


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
        return 2
    rce = argv[1]
    outdir = argv[2] if len(argv) == 3 else None
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    man = tomllib.load(open(os.path.join(CORPUS, "manifest.toml"), "rb"))
    counts = {"MATCH": 0, "MATCH-RESOURCE": 0, "DIFFERS": 0, "KEPT-RESOURCE": 0, "REQUIRES-CHANGE": 0, "INTERNAL": 0}
    by_rule, first_rule, only_rule = {}, {}, {}
    rows = []
    tmp = os.environ.get("TMPDIR", "/tmp")
    for c in man["case"]:
        item = c["provenance"]["item"]
        src = open(os.path.join(CORPUS, c["file"]), "rb").read()
        exp = json.load(open(os.path.join(CORPUS, c["expected"])))
        text, diags, _ = lift_py.lift(item + ".py", src)
        if diags:
            rules = sorted({d[1] for d in diags})
            for r in rules:
                by_rule[r] = by_rule.get(r, 0) + 1
            fr = min(diags)[1]
            first_rule[fr] = first_rule.get(fr, 0) + 1
            if len(rules) == 1:
                only_rule[rules[0]] = only_rule.get(rules[0], 0) + 1
            counts["REQUIRES-CHANGE"] += 1
            rows.append(f"REQUIRES-CHANGE {item}: {', '.join(rules)}")
            continue
        path = os.path.join(outdir or tmp, item.replace("/", "__") + ".rce")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        chk = subprocess.run([rce, "check", path], capture_output=True)
        if chk.returncode != 0:
            counts["INTERNAL"] += 1
            msg = chk.stdout.decode(errors="replace").strip().replace(path, item + ".rce")
            rows.append(f"INTERNAL {item}: {msg[:300]}")
            continue
        status, out, cls, stopped = observe(rce, path)
        if status is None:
            counts["INTERNAL"] += 1
            rows.append(f"INTERNAL {item}: {stopped}")
            continue
        want = bytes.fromhex(exp["stdout_hex"])
        if stopped is not None:
            res = any(k in stopped for k in RESOURCE_TRAPS)
            if c["class"] == "resource" and res:
                counts["MATCH-RESOURCE"] += 1
                rows.append(f"MATCH-RESOURCE {item}: {stopped}")
            elif res:
                counts["KEPT-RESOURCE"] += 1
                rows.append(f"KEPT-RESOURCE {item}: {stopped}")
            else:
                counts["DIFFERS"] += 1
                rows.append(f"DIFFERS {item}: the lift stopped ({stopped}); oracle status {exp['status']}, class {exp['stderr_class']!r}")
            continue
        if status != exp["status"]:
            counts["DIFFERS"] += 1
            rows.append(f"DIFFERS {item}: status oracle {exp['status']}, lift {status}")
        elif out != want:
            counts["DIFFERS"] += 1
            rows.append(f"DIFFERS {item}: stdout {first_diff(want, out)}")
        elif cls != exp["stderr_class"]:
            counts["DIFFERS"] += 1
            rows.append(f"DIFFERS {item}: stderr class oracle {exp['stderr_class']!r}, lift {cls!r}")
        else:
            counts["MATCH"] += 1
            key = hashlib.sha256(text.encode()).hexdigest()
            rows.append(f"MATCH {item} lifted {key}")
    for r in rows:
        print(r)
    print()
    print(f"cases {len(man['case'])}")
    for k, v in counts.items():
        print(f"{k} {v}")
    print("requires change, cases naming each rule (a case can name several):")
    for k in sorted(by_rule):
        print(f"  {k} {by_rule[k]}")
    print("requires change, by the rule of the first line reported:")
    for k in sorted(first_rule):
        print(f"  {k} {first_rule[k]}")
    print("requires change, cases whose only rule is:")
    for k in sorted(only_rule):
        print(f"  {k} {only_rule[k]}")
    return 0 if counts["INTERNAL"] == 0 and counts["DIFFERS"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
