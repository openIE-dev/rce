# SPDX-License-Identifier: MIT OR Apache-2.0
# Conversion to and from Roman numerals, checked by a round trip.
VALS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
DIGIT = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def to_roman(n):
    out = []
    for v, s in VALS:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


def from_roman(s):
    total = 0
    for i, ch in enumerate(s):
        v = DIGIT[ch]
        if i + 1 < len(s) and DIGIT[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


for n in [1, 4, 9, 14, 40, 90, 400, 1994, 2024, 3999]:
    r = to_roman(n)
    print(n, r, from_roman(r))
print(sum(1 for n in range(1, 4000) if from_roman(to_roman(n)) != n))
