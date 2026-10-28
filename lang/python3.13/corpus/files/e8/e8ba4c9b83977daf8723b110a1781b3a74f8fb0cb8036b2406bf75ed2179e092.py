# SPDX-License-Identifier: MIT OR Apache-2.0
# Recursion with no base case reaches the interpreter's recursion limit,
# which raises RecursionError; here it is not caught.


def down(n):
    return down(n + 1) + 1


print("start")
print(down(0))
