# SPDX-License-Identifier: MIT OR Apache-2.0
calls = []


def log(x):
    calls.append(x)


log(1)
print(calls)
