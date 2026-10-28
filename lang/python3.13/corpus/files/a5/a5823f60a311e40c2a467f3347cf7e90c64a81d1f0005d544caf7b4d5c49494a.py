# SPDX-License-Identifier: MIT OR Apache-2.0
# The longest common subsequence by dynamic programming, with one
# subsequence recovered from the table.


def lcs(a, b):
    m, n = len(a), len(b)
    t = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                t[i][j] = t[i - 1][j - 1] + 1
            else:
                t[i][j] = max(t[i - 1][j], t[i][j - 1])
    out = []
    i, j = m, n
    while i and j:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1])
            i -= 1
            j -= 1
        elif t[i - 1][j] >= t[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return t[m][n], "".join(reversed(out))


for a, b in [("ABCBDAB", "BDCABA"), ("AGGTAB", "GXTXAYB"), ("", "abc"), ("same", "same"), ("kitten", "sitting")]:
    print(repr(a), repr(b), *lcs(a, b))
