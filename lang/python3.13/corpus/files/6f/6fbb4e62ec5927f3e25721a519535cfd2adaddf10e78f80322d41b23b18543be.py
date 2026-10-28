# SPDX-License-Identifier: MIT OR Apache-2.0
# Binary search over a sorted list, for present and absent values, checked
# against bisect.
import bisect


def search(xs, target):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo, lo < len(xs) and xs[lo] == target


xs = [i * i - 100 for i in range(50)]
for t in [-100, -99, 0, 1, 44, 2301, 2302, 5000, -1000]:
    i, ok = search(xs, t)
    print(t, i, ok, i == bisect.bisect_left(xs, t))
