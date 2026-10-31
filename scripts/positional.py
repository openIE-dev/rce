"""Positional command lines for the repository's Python scripts
(spec/proposals/positional-cli.md, PC-1 to PC-8).

An invocation is `SCRIPT WORD... OPERAND...`: the script's words, in any
order among themselves, then its operands in order. A word with values takes
the next argument(s) as its values. No argument starts with `-` (PC-6): a
removed flag prints one line that names its new spelling and the script
exits 2 without running (decision 2 of section 5).

    from positional import read
    w, ops = read("scripts/x.py", {"jobs": 1, "quiet": 0}, "OUT", 1, 1,
                  removed={"--out": 0})
"""
import sys


def _fail(line):
    sys.stderr.write(line + "\n")
    sys.exit(2)


def read(prog, words, operands, lo, hi=None, argv=None, removed=None, many=(), fill=None):
    """Read `argv` (default: sys.argv[1:]) against the script's words.

    words: name -> number of values (0 or 1); a name in `many` may repeat.
    operands: the usage text of the operands; lo and hi bound their number
    (hi None: no bound). removed: an old flag -> the operand position its
    value moves to (`--log D` -> D); any other `--NAME` becomes the word
    NAME. fill: an operand position -> the text the new spelling shows there
    when a later operand was given and this one was not. Returns (dict of word -> value, True, or a list for `many`;
    list of operands).
    """
    a = list(sys.argv[1:] if argv is None else argv)
    usage = "usage: " + " ".join([prog] + [f"[{w}{' V' * n}]{'...' if w in many else ''}" for w, n in words.items()] + ([operands] if operands else []))
    removed = removed or {}
    if any(x.startswith("-") for x in a):
        first = next(x for x in a if x.startswith("-")).split("=")[0]
        ws, ops, placed, i = [], [], [], 0
        while i < len(a):
            x = a[i]
            if not x.startswith("-"):
                ops.append(x)
                i += 1
                continue
            f, eq, v = x.partition("=")
            name = f.lstrip("-")
            if f in removed:
                val = v if eq else (a[i + 1] if i + 1 < len(a) else "")
                placed.append((removed[f], val))
                i += 1 if eq else 2
                continue
            if name not in words:
                _fail(f"`{x}`: no argument starts with `-` (PC-6); {usage}")
            ws.append(name)
            if words[name]:
                ws.append(v if eq else (a[i + 1] if i + 1 < len(a) else ""))
                i += 1 if eq else 2
            else:
                i += 1
        for k, v in sorted(placed):
            while len(ops) < k and (fill or {}).get(len(ops)) is not None:
                ops.append(fill[len(ops)])
            ops.insert(min(k, len(ops)), v)
        _fail(f"{first} was removed; write: {' '.join([prog] + ws + ops)}")
    got, i = {}, 0
    while i < len(a) and a[i] in words:
        w, n = a[i], words[a[i]]
        if n and i + n >= len(a):
            _fail(f"`{w}` takes a value; {usage}")
        v = a[i + 1] if n else True
        if w in many:
            got.setdefault(w, []).append(v)
        elif w in got:
            _fail(f"the word `{w}` is given twice; {usage}")
        else:
            got[w] = v
        i += 1 + n
    ops = a[i:]
    if len(ops) < lo or (hi is not None and len(ops) > hi):
        _fail(usage)
    return got, ops
