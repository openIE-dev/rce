# SPDX-License-Identifier: MIT OR Apache-2.0
# Prime factorisation by trial division, including numbers past 2^64.


def factors(n):
    fs = []
    p = 2
    while p * p <= n:
        while n % p == 0:
            fs.append(p)
            n //= p
        p += 1 if p == 2 else 2
    if n > 1:
        fs.append(n)
    return fs


for n in [1, 2, 12, 97, 360, 1001, 65536, 600851475143, 9999999967, 2 ** 64 - 1, 2 ** 64 + 1, 10 ** 20 + 1]:
    print(n, factors(n))
