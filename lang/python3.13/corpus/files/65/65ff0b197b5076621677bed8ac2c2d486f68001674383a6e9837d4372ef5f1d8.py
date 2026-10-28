# SPDX-License-Identifier: MIT OR Apache-2.0
# How print shows values of the built-in types: str at the top level, repr
# inside containers.
print(42, -7, 3.5, True, None, "text", b"bytes", 2 + 3j)
print([1, 2, 3], (1,), (), [], {}, set(), {1: "a", "b": [2, 3]})
print(["a", 'b"c', "d'e", "f\ng"], ("x", None, True))
print([1.0, 0.1, 1e100, -0.0, float("inf"), float("nan")])
print({"k": {"n": [1, {"m": ()}]}})
print(range(5), range(1, 10, 3), list(range(3)))
print(frozenset(), bytearray(b"ab"), memoryview(b"x").tobytes())
print("a", "b", 1, 2, sep="-", end="!\n")
print()
print("no newline", end="")
print(" then newline")
print(*[1, 2, 3])
print(str(None), repr("q"), repr(1.0), str(1.0), repr([None]))
