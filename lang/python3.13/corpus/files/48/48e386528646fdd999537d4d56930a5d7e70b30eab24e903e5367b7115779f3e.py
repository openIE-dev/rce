# SPDX-License-Identifier: MIT OR Apache-2.0
# A binary min-heap written by hand, and heapq on the same data.
import heapq


class MinHeap:
    def __init__(self):
        self.a = []

    def push(self, x):
        a = self.a
        a.append(x)
        i = len(a) - 1
        while i > 0 and a[(i - 1) // 2] > a[i]:
            p = (i - 1) // 2
            a[p], a[i] = a[i], a[p]
            i = p

    def pop(self):
        a = self.a
        top = a[0]
        last = a.pop()
        if a:
            a[0] = last
            i = 0
            while True:
                l, r, m = 2 * i + 1, 2 * i + 2, i
                if l < len(a) and a[l] < a[m]:
                    m = l
                if r < len(a) and a[r] < a[m]:
                    m = r
                if m == i:
                    break
                a[m], a[i] = a[i], a[m]
                i = m
        return top


data = [9, 4, 7, 1, 8, 2, 2, 6, 3, 5, 0]
h = MinHeap()
for d in data:
    h.push(d)
print([h.pop() for _ in data])
q = list(data)
heapq.heapify(q)
print(q, [heapq.heappop(q) for _ in data])
print(heapq.nlargest(3, data), heapq.nsmallest(3, data), list(heapq.merge([1, 4, 9], [2, 3, 10])))
