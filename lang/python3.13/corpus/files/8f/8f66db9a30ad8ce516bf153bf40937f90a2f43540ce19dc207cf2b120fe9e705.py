# SPDX-License-Identifier: MIT OR Apache-2.0
# The sieve of Eratosthenes: the primes below 200 and the count below 100000.


def sieve(n):
    composite = bytearray(n)
    i = 2
    while i * i < n:
        if not composite[i]:
            composite[i * i::i] = b"\x01" * len(range(i * i, n, i))
        i += 1
    return [k for k in range(2, n) if not composite[k]]


print(*sieve(200))
print(len(sieve(100000)))
