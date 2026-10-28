# SPDX-License-Identifier: MIT OR Apache-2.0
# An uncaught exception ends the program with exit status 1, after what was
# printed and after finally blocks.


def check(n):
    if n > 2:
        raise ValueError(f"n too large: {n}")
    print("ok", n)


try:
    for i in range(5):
        check(i)
finally:
    print("finally runs")
