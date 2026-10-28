# SPDX-License-Identifier: MIT OR Apache-2.0
# Methods of str on ASCII text.
s = "  The quick brown fox  "
print(f"[{s.strip()}]", f"[{s.lstrip()}]", f"[{s.rstrip()}]")
print(s.upper(), s.lower(), s.title(), s.swapcase(), "hello world".capitalize())
print(s.split(), "a,b,,c".split(","), "a,b,c,d".split(",", 2), "a b  c".split(" "), "x".split(","))
print("a,b,c".rsplit(",", 1), "line1\nline2\r\nline3".splitlines(), "".split(","))
print("-".join(["x", "y", "z"]), "".join([]), ", ".join("abc"))
print("aaaa".replace("a", "b", 2), "banana".replace("an", "AN"), "abc".replace("", "-"))
print("seafood".startswith("sea"), "golang".endswith(("ng", "x")), "chicken".find("ken"), "chicken".find("dmr"))
print("go gopher".rfind("go"), "cheese".count("e"), "five".count(""))
print("xxhixx".strip("x"), "xxhixx".lstrip("x"), "xxhixx".removeprefix("x"), "xxhixx".removesuffix("xx"))
print("42".zfill(5), "-42".zfill(5), "ab".center(6, "*"), "ab".ljust(4) + "|", "ab".rjust(4))
print("a1".isalnum(), "abc".isalpha(), "123".isdigit(), " \t".isspace(), "Title Case".istitle(), "ABC".isupper())
print("a-b-c".partition("-"), "a-b-c".rpartition("-"), "abc".partition("x"))
print("hello"[::-1], sorted("hello"), "%s=%d" % ("x", 5), "{}+{}={}".format(1, 2, 3))
