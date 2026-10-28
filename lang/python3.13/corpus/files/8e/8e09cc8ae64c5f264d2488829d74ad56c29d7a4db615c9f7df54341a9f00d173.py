# SPDX-License-Identifier: MIT OR Apache-2.0
# Strings are sequences of code points: len counts code points, and the
# UTF-8 and UTF-16 encodings have their own lengths.
words = ["abc", "héllo", "日本語", "😀", "e\u0301", "👍🏽", "é", "👨\u200d👩\u200d👧"]
for w in words:
    print(w, len(w), len(w.encode("utf-8")), len(w.encode("utf-16-le")) // 2, [hex(ord(c)) for c in w])
for i, c in enumerate("aé日😀"):
    print(i, ord(c), c, len(c.encode()))
print("日本".encode("utf-8"), "日本".encode("utf-8").hex(" "))
print(b"\xe6\x97\xa5".decode("utf-8"), b"\xff".decode("utf-8", "replace"), b"ab\xffcd".decode("utf-8", "ignore"))
print(bytes([104, 105]).decode("ascii"), "é".encode("latin-1"), "€".encode("cp1252"))
