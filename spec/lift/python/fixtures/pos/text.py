# SPDX-License-Identifier: MIT OR Apache-2.0
# Text of code points: indexing, slices, methods, repr and format specs.
s = "hello, world"
t = "héllo, wörld"
print(t, len(t), t[1], t[-1], t[2:5], t[::-1], t[::3])
print(s, len(s), s[1], s[-1], s[2:5], s[::-1], s[::3])
print(s.upper() if s.isalpha() else "not alpha", "abc".isalpha(), "ab1".upper(), "a,b,,c".split(","), " x  y ".split())
print("-".join(["x", "y", "z"]), "banana".replace("an", "AN"), "chicken".find("ken"), "five".count(""))
print(repr("it's"), repr('say "hi"'), repr("tab\there\x00​😀"), ord("é"), chr(9731))
for w in ["go", "golang"]:
    print(f"[{w:>8}] [{w:<8}] [{w:*^8}] [{w!r:>10}] [{len(w):03d}] [{255:#x}] [{1234567:,}]")
word = ""
for i in range(5):
    word += chr(ord("a") + i)
print(word, "ell" in "hello", "z" not in "hello", "ab" * 3)
