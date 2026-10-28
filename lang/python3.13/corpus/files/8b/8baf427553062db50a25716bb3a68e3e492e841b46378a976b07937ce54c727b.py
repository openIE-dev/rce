# SPDX-License-Identifier: MIT OR Apache-2.0
# RecursionError can be caught; recursion below the default limit of 1000
# frames completes.
import sys


def depth(n):
    return 0 if n == 0 else 1 + depth(n - 1)


def forever(n):
    return forever(n + 1)


print(sys.getrecursionlimit(), depth(900))
try:
    forever(0)
except RecursionError as e:
    print("caught", type(e).__name__, str(e)[:30])
print(depth(500))
