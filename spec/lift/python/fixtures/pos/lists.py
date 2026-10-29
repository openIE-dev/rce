# SPDX-License-Identifier: MIT OR Apache-2.0
# Lists and dictionaries: mutation through a function, insertion order,
# sorting, comprehensions, unpacking and loops with else.
def push_squares(xs, n):
    for i in range(n):
        xs.append(i * i)


def swap(a, i, j):
    a[i], a[j] = a[j], a[i]


xs = []
push_squares(xs, 6)
swap(xs, 0, 5)
print(xs, xs[1:3], sorted(xs), sorted(xs, reverse=True), xs.index(9), xs.count(4))
xs.insert(1, 99)
xs.pop()
print(xs, [x for x in xs if x % 2 == 0], list(range(10, 0, -3)))
counts = {}
for ch in "mississippi":
    counts[ch] = counts.get(ch, 0) + 1
print(counts, len(counts), "s" in counts, list(counts.items()))
grid = [[0] * 3 for _ in range(3)]
grid[1][2] = 5
for r, row in enumerate(grid):
    print(r, row)
for n in range(2, 12):
    for d in range(2, n):
        if n % d == 0:
            break
    else:
        print(n, end=" ")
print()
k = 0
while k < 10:
    k += 3
print(k, *[1, 2, 3], sep="-")
