# SPDX-License-Identifier: MIT OR Apache-2.0
# Quicksort with Lomuto partitioning on a sequence from a fixed linear
# congruential generator (arithmetic modulo 2^32).


def quicksort(a, lo, hi):
    if hi - lo < 2:
        return
    p = a[hi - 1]
    i = lo
    for j in range(lo, hi - 1):
        if a[j] < p:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi - 1] = a[hi - 1], a[i]
    quicksort(a, lo, i)
    quicksort(a, i + 1, hi)


s = 12345
a = []
for _ in range(40):
    s = (s * 1103515245 + 12345) % 2 ** 32
    a.append((s >> 16) % 1000)
print(a)
quicksort(a, 0, len(a))
print(a)
print(a == sorted(a))
