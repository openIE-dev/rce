# SPDX-License-Identifier: MIT OR Apache-2.0
# repr of a float is the shortest text that reads back as the same value;
# it uses an exponent below 1e-4 and from 1e16 on.
vals = [0.1, 0.2, 0.30000000000000004, 1e16, 9999999999999998.0, 1e-4, 1e-5, 1e22, 1e23, 5e-324, 1.7976931348623157e308,
        float("inf"), float("-inf"), float("nan"), -0.0, 1.0, 100.0, 1e15 + 0.3, 2.0 ** 53, 2.0 ** 53 + 1, 0.1 + 0.7, 1 / 3]
for v in vals:
    print(repr(v), str(v), float(repr(v)) == v or v != v)
print(0.1 + 0.2 == 0.3, round(0.1 + 0.2, 15) == 0.3, 1.1 * 3, 4.35 * 100, 0.57 * 100)
print(float.fromhex("0x1.8p1"), (0.1).hex(), (1.0).as_integer_ratio(), (0.1).as_integer_ratio())
print(float("1e400"), float("-1e-400"), float("0x1p3") if False else float.fromhex("0x1p-1074"))
