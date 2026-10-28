# SPDX-License-Identifier: MIT OR Apache-2.0
# CPython limits conversion between int and decimal text to 4300 digits by
# default (sys.get_int_max_str_digits): a longer conversion raises
# ValueError, caught twice here and uncaught at the end.
import sys

print(sys.get_int_max_str_digits())
ok = 10 ** 4299
print(len(str(ok)), str(ok)[:5])
try:
    str(10 ** 4300)
except ValueError as e:
    print("ValueError", str(e)[:40])
try:
    int("1" * 4301)
except ValueError as e:
    print("ValueError", str(e)[:40])
print(hex(10 ** 5000)[:12], len(hex(10 ** 5000)))
big = 10 ** 5000
print(big % 1000007, big.bit_length())
print(str(big))
