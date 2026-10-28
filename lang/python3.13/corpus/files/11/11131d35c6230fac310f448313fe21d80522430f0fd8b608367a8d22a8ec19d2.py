# SPDX-License-Identifier: MIT OR Apache-2.0
# Knuth-Morris-Pratt substring search, checked against str.find.


def failure(p):
    f = [0] * len(p)
    k = 0
    for i in range(1, len(p)):
        while k and p[i] != p[k]:
            k = f[k - 1]
        if p[i] == p[k]:
            k += 1
        f[i] = k
    return f


def find_all(text, p):
    if not p:
        return []
    f, out, k = failure(p), [], 0
    for i, ch in enumerate(text):
        while k and ch != p[k]:
            k = f[k - 1]
        if ch == p[k]:
            k += 1
        if k == len(p):
            out.append(i - len(p) + 1)
            k = f[k - 1]
    return out


print(failure("ababaca"), failure("aaaa"))
text = "abababcabababcababab"
for p in ["abab", "ababc", "c", "x", text]:
    found = find_all(text, p)
    print(p, found, (found[0] if found else -1) == text.find(p))
