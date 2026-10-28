# SPDX-License-Identifier: MIT OR Apache-2.0
# All permutations of four letters in lexicographic order by the
# next-permutation step, checked against itertools.permutations.
import itertools


def next_perm(a):
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i < 0:
        return False
    j = len(a) - 1
    while a[j] <= a[i]:
        j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1:] = reversed(a[i + 1:])
    return True


a = list("abcd")
seen = ["".join(a)]
while next_perm(a):
    seen.append("".join(a))
for k in range(0, len(seen), 6):
    print(" ".join(seen[k:k + 6]))
print(len(seen), seen == ["".join(p) for p in itertools.permutations("abcd")])
b = list("aabb")
m = 1
while next_perm(b):
    m += 1
print(m, "".join(b))
