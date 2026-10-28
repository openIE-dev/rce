# SPDX-License-Identifier: MIT OR Apache-2.0
# for over ranges and sequences, while, nested loops and the else clause.
for i in range(3):
    print(i, end=" ")
print()
n = 1
while n < 100:
    n *= 3
print(n)
for i in range(1, 4):
    print(" ".join(str(i * j) for j in range(1, 4)))
print(list(range(10, 0, -3)), list(range(0)), list(range(5, 2)), list(range(-3, 3, 2)))
for i, c in enumerate("go!"):
    print(f"{i}:{c}", end=" ")
print()
for a, b in zip([1, 2, 3], "xy"):
    print(a, b, end="; ")
print()
for k in range(3):
    pass
else:
    print("loop finished", k)
count = 0
while True:
    count += 1
    if count >= 4:
        break
print(count)
