# SPDX-License-Identifier: MIT OR Apache-2.0
# Format specifications at their edges: grouping, signs, alignment of
# special values, precision on strings and large widths.
for v in [0.0, -0.0, 1e100, float("inf"), float("nan"), -1234567.891]:
    print(f"[{v:>12}] [{v:<+12,}] [{v:^12.3}] [{v:012}]")
for n in [0, -7, 2 ** 70]:
    print(f"[{n:>12}] [{n:<+12,}] [{n:^12}] [{n:012}] [{n:012.3f}]")
print(f"{12345678:,d} {12345678:_x} {255:#010b} {-255:#x} {255:=+8}")
print(f"{'abc':.1} {'abc':>5.2} {'é':^5} {'日本':*<6}")
print(f"{0.000012345:.3g} {123456789:.3g} {0.5:.0%} {1 / 3:.3%} {1e-7:f} {1e22:f}")
print(format(10 ** 30, "e"), format(10 ** 30, ".3e"), format(float(10 ** 30), ".0f"), format(-1e-300, ".2g"))
print("{:,}".format(10 ** 25), "{:n}".format(1234567), "{!s:>4}|{!r:<6}|".format(1, "a"))
try:
    format(1.5, "d")
except ValueError as e:
    print("ValueError", e)
try:
    format("s", "d")
except ValueError as e:
    print("ValueError", e)
