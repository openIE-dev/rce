# SPDX-License-Identifier: MIT OR Apache-2.0
# break and continue, and the else clause of for and while, which runs only
# when the loop ends without break.
for i in range(10):
    if i % 2 == 1:
        continue
    if i > 6:
        break
    print(i, end=" ")
print()
for n in range(2, 20):
    for d in range(2, n):
        if n % d == 0:
            break
    else:
        print(n, end=" ")
print()
found = None
for r, row in enumerate([[1, 2], [3, 4], [5, 6]]):
    if 4 in row:
        found = r
        break
else:
    found = -1
print(found)
k = 0
while k < 3:
    k += 1
else:
    print("while else", k)
