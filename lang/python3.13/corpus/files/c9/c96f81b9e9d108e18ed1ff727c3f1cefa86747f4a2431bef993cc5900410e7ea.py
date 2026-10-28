# SPDX-License-Identifier: MIT OR Apache-2.0
# Integer literals in every base, digit separators, and conversion to and
# from text in bases.
print(255, 0xFF, 0o377, 0b11111111, 1_000_000, 0x_FF_FF)
n = 48879
print(hex(n), oct(n), bin(n), hex(-n), bin(-5), f"{n:x} {n:o} {n:b}")
print(int("ff", 16), int("-0x1f", 0), int("0o17", 0), int("0b101", 0), int("777", 8), int("zz", 36), int(" 42 "), int("1_000"))
print(int(3.99), int(-3.99), int(True), int("-0"), float("1e3"), float(" -1.5 "), float("inf"), float("-nan"))
try:
    int("1.5")
except ValueError as e:
    print("ValueError", e)
try:
    int("08", 0)
except ValueError as e:
    print("ValueError", e)
print(1.5e3, 0x10 + 0o10 + 0b10, 1e-3, 1_0.5, 10j, 0.1j * 1j)
print(n.bit_length(), n.bit_count(), (255).to_bytes(2, "big"), int.from_bytes(b"\x01\x00", "little"))
