# SPDX-License-Identifier: MIT OR Apache-2.0
def adder(n):
    return lambda x: x + n


print(adder(2)(3))
