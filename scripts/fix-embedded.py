#!/usr/bin/env python3
"""Run fix-hygiene over the rce programs embedded as string literals (raw or plain without escapes) in Rust test files.
usage: fix-embedded.py RCE_BIN FILE.rs..."""
import re, subprocess, sys, tempfile, os
rce = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
def literals(s):
    i = 0; n = len(s)
    while i < n:
        c = s[i]
        if s.startswith("//", i):
            i = s.find("\n", i); i = n if i < 0 else i; continue
        m = re.compile(r"r(#*)\"").match(s, i) if c == "r" and (i == 0 or not (s[i-1].isalnum() or s[i-1] == "_")) else None
        if m:
            end = s.find("\"" + m.group(1), m.end())
            if end < 0: return
            yield m.end(), end; i = end + 1 + len(m.group(1)); continue
        if c == "\"":
            j = i + 1
            while j < n and s[j] != "\"":
                j += 2 if s[j] == "\\" else 1
            yield i + 1, j; i = j + 1; continue
        if c == "\x27":
            m = re.compile(r"\x27(\\\\.[^\x27]*|[^\x27\\\\])\x27").match(s, i)
            i += len(m.group(0)) if m else 1; continue
        i += 1
for path in sys.argv[2:]:
    s = open(path).read(); out = []; last = 0; n = 0
    for a, b in literals(s):
        body = s[a:b]
        if "fn " not in body or "\n" not in body or "\\" in body: continue
        with tempfile.NamedTemporaryFile("w", suffix=".rce", delete=False) as t:
            t.write(body); name = t.name
        subprocess.run([sys.executable, os.path.join(here, "fix-hygiene.py"), rce, name], capture_output=True)
        new = open(name).read(); os.unlink(name)
        if new != body: n += 1; out.append(s[last:a]); out.append(new); last = b
    out.append(s[last:])
    if n: open(path, "w").write("".join(out))
    print(path, n, "programs changed")
