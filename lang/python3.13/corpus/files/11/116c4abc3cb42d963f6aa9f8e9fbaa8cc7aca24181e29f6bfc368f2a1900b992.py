# SPDX-License-Identifier: MIT OR Apache-2.0
# Concatenation, length in code points, indexing, slicing and immutability.
s = "hello"
t = s + ", " + "world"
print(t, len(t))
print(s[0], s[1:3], s[:2], s[3:], s[:], s[-1], s[-3:], s[::-1], s[::2])
print(s * 3, 3 * "ab", "-" * 0)
print("ell" in s, "z" not in s, s.find("l"), s.index("o"), s.count("l"))
try:
    s[0] = "j"
except TypeError as e:
    print("TypeError", e)
print("j" + s[1:])
u = ""
for i in range(5):
    u += chr(ord("a") + i)
print(u, ord("A"), chr(9731), "\N{SNOWMAN}" == chr(9731))
print("tab\tquote\"back\\slash", r"raw\n", "\x41é\U0001F600")
print("con" "cat", """multi
line""")
