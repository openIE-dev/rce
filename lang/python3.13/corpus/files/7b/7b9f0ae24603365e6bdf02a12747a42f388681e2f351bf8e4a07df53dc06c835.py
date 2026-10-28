# SPDX-License-Identifier: MIT OR Apache-2.0
# Strings compare by code point, so U+FFFF sorts before U+10000, unlike
# languages whose strings are UTF-16 code units.
pairs = [("\uffff", "\U00010000"), ("é", "f"), ("Z", "a"), ("abc", "abd"), ("ab", "abc"), ("\x00", ""), ("é", "e\u0301")]
for a, b in pairs:
    print(ascii(a), ascii(b), a < b, a == b)
ws = ["\U0001F600", "\uffff", "zebra", "Zebra", "éclair", "eclair", "", "10", "9"]
print(ascii(sorted(ws)))
print(max("abc"), min(["b", "B", "a"]), sorted(["a", "B", "b", "A"], key=str.casefold))
print(b"abc" < b"abd", b"\xff" > b"\x00", bytes([255]) > "ÿ".encode("utf-8"))
