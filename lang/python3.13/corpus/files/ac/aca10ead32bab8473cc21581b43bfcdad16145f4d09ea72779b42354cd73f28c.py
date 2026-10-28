# SPDX-License-Identifier: MIT OR Apache-2.0
# bool is a subclass of int: True is 1 and False is 0 in arithmetic, as
# dictionary keys and in sums.
print(True + True, True * 10, -True, ~True, True / 2, True // 2, True ** 0)
print(isinstance(True, int), issubclass(bool, int), type(True + 0), True == 1, False == 0)
print(sum([True, False, True, True]), max(True, 2), [10, 20][True], "ab" * True)
d = {1: "int one", True: "bool true", 1.0: "float one", 0: "zero", False: "false"}
print(d, len(d))
print({True, 1, 1.0}, {0, False, 0.0, -0.0})
print(hash(True) == hash(1), hash(1.0) == hash(1), str(True), repr(False), int(True), float(False))
print(bool("False"), bool(0j), True & False, True | False, True ^ True, True << 3)
