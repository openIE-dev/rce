# SPDX-License-Identifier: MIT OR Apache-2.0
# Levenshtein distance between strings, counted over code points.


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


for a, b in [("kitten", "sitting"), ("flaw", "lawn"), ("", "abc"), ("abc", ""), ("same", "same"), ("héllo", "hello"), ("日本語", "日本"), ("intention", "execution")]:
    print(a, b, lev(a, b))
