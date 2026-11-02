#!/usr/bin/env python3
"""Reference results of E4's kernels, written from each kernel's comment, not from its code's output.

The expected value of run(n) for every declared input comes from here; the
native programs and the WebAssembly modules under test are compared with it
(PROTOCOL.md, "Equivalence evidence"). All arithmetic is modulo 2^32.
"""
M = 0xFFFFFFFF


def sumsq(n):
    s = 0
    for i in range(n):
        s = (s * 31 + i * i) & M
    return s


def collatz(n):
    total = 0
    for v in range(1, n + 1):
        x, steps = v, 0
        while x != 1 and steps < 1000:
            x = (3 * x + 1) & M if x & 1 else x >> 1
            steps += 1
        total = (total + steps) & M
    return total


def sieve(n):
    n = min(n, 8192)
    composite = bytearray(max(n, 2))
    count = 0
    for i in range(2, n):
        if not composite[i]:
            count += 1
            for j in range(i * i, n, i):
                composite[j] = 1
    return count


def sort(n):
    n = min(n, 1024)
    x, a = 12345, []
    for _ in range(n):
        x = (x * 1103515245 + 12345) & M
        a.append((x >> 16) & 0x7FFF)
    a.sort()
    return sum(v * (i + 1) for i, v in enumerate(a)) & M


def crc32(n):
    crc, x = 0xFFFFFFFF, 1
    for _ in range(n):
        x = (x * 1664525 + 1013904223) & M
        crc ^= x >> 24
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 if crc & 1 else 0)
    return ~crc & M


def matmul(n):
    n = min(n, 24)
    A = [[i * j + 1 for j in range(n)] for i in range(n)]
    B = [[i + 2 * j + 3 for j in range(n)] for i in range(n)]
    total = 0
    for i in range(n):
        for j in range(n):
            c = sum(A[i][k] * B[k][j] for k in range(n)) & M
            total += c * (i + j + 1)
    return total & M


def fib(n):
    n = min(n, 30)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, (a + b) & M
    return a


def popcount(n):
    x, total = 7, 0
    for _ in range(n):
        x = (x * 22695477 + 1) & M
        total += bin(x).count("1")
    return total & M


KERNELS = {f.__name__: f for f in (sumsq, collatz, sieve, sort, crc32, matmul, fib, popcount)}
