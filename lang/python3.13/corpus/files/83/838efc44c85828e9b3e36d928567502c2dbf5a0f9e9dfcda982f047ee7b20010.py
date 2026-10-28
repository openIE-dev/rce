# SPDX-License-Identifier: MIT OR Apache-2.0
# if, elif and else, conditional expressions and truth values.


def classify(n):
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    elif n % 2 == 0:
        return "even"
    else:
        return "odd"


for n in [-5, 0, 4, 7]:
    print(n, classify(n))
grade = 83
print("A" if grade >= 90 else "B" if grade >= 80 else "C")
for v in [0, 1, "", "x", [], [0], None]:
    if v:
        print(repr(v), "truthy")
    else:
        print(repr(v), "falsy")
if (y := 10 * 3) > 20:
    print("big", y)
