# SPDX-License-Identifier: MIT OR Apache-2.0
def count(n):
    i = 0
    while i < n:
        yield i
        i += 1


print(list(count(3)))
