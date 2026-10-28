# SPDX-License-Identifier: MIT OR Apache-2.0
# A stable merge sort of records by key; equal keys keep their order.


def mergesort(a):
    if len(a) <= 1:
        return a
    m = len(a) // 2
    left, right = mergesort(a[:m]), mergesort(a[m:])
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if right[j][0] < left[i][0]:
            out.append(right[j])
            j += 1
        else:
            out.append(left[i])
            i += 1
    return out + left[i:] + right[j:]


data = [(3, "a"), (1, "b"), (2, "c"), (3, "d"), (1, "e"), (2, "f"), (0, "g"), (3, "h")]
print(mergesort(data))
print(mergesort(data) == sorted(data, key=lambda r: r[0]))
print(data)
