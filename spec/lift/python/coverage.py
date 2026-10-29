#!/usr/bin/env python3
"""Function-level coverage of the Python lift (section 16: the frontier is
drawn at function granularity).

Usage (positional arguments only):

    coverage.py RCE FILE

One line per function of FILE: `lifted NAME` when the function and every
function it calls lift and the printed RCE text passes `rce check`
(`internal NAME` when the lift applied and the check failed: a lifter
defect), or `requires change NAME: RULE, ...` with every rule named. Then
the counts: functions, lifted, the share, and per rule the functions that
name it, the functions whose first line reported names it, and the
functions whose only rule it is (L-PY-CALLEE aside).
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lift_py  # noqa: E402


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
        return 2
    rce, path = argv[1], argv[2]
    with open(path, "rb") as f:
        src = f.read()
    name = os.path.basename(path)
    res = lift_py.lift_functions(name, src)
    lifted, internal = 0, 0
    by, first, only = {}, {}, {}
    with tempfile.TemporaryDirectory() as d:
        for fn, text, diags in res:
            if diags:
                rules = sorted({x[1] for x in diags})
                for r in rules:
                    by[r] = by.get(r, 0) + 1
                fr = min(diags)[1]
                first[fr] = first.get(fr, 0) + 1
                own = [r for r in rules if r != "L-PY-CALLEE"]
                if len(own) == 1:
                    only[own[0]] = only.get(own[0], 0) + 1
                print(f"requires change {fn}: {', '.join(rules)}")
                continue
            p = os.path.join(d, fn + ".rce")
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            chk = subprocess.run([rce, "check", p], capture_output=True, text=True)
            if chk.returncode != 0:
                internal += 1
                print(f"internal {fn}: {chk.stdout.strip().replace(p, fn + '.rce')[:300]}")
            else:
                lifted += 1
                print(f"lifted {fn}")
    n = len(res)
    print()
    print(f"module {name}")
    print(f"functions {n}")
    print(f"lifted {lifted}")
    print(f"internal {internal}")
    print(f"coverage {100.0 * lifted / n:.1f}%" if n else "coverage n/a")
    print("requires change, functions naming each rule:")
    for k in sorted(by):
        print(f"  {k} {by[k]}")
    print("requires change, by the rule of the first line reported:")
    for k in sorted(first):
        print(f"  {k} {first[k]}")
    print("requires change, functions whose only rule (L-PY-CALLEE aside) is:")
    for k in sorted(only):
        print(f"  {k} {only[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
