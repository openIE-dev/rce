# SPDX-License-Identifier: MIT OR Apache-2.0
# FNV-1a hashes in 32 and 64 bits; Python integers do not wrap, so each
# product is reduced with a mask.


def fnv32a(data):
    h = 2166136261
    for b in data:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def fnv64a(data):
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


for s in ["", "a", "foobar", "hello, world", "日本"]:
    d = s.encode("utf-8")
    print(repr(s), f"{fnv32a(d):08x}", f"{fnv64a(d):016x}")
h = 2166136261
for b in b"foobar":
    h = (h ^ b) * 16777619
print(h.bit_length(), h & 0xFFFFFFFF == fnv32a(b"foobar"))
