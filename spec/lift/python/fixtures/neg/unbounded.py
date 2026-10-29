# SPDX-License-Identifier: MIT OR Apache-2.0
def search(xs, t):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


print(search([1, 3, 5, 7], 5))
n = 1
while True:
    n += 1
    if n > 5:
        break
print(n)
