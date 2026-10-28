# SPDX-License-Identifier: MIT OR Apache-2.0
# Run-length encoding and decoding over code points.
import itertools
import re


def encode(s):
    return "".join(f"{len(list(g))}{ch}" for ch, g in itertools.groupby(s))


def decode(s):
    return "".join(ch * int(n) for n, ch in re.findall(r"(\d+)(\D)", s))


for s in ["", "a", "aaabccddddde", "WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWBWWWWWWWWWWWWWW", "ééé日日x"]:
    e = encode(s)
    print(repr(s), repr(e), decode(e) == s)
