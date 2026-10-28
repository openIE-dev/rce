# SPDX-License-Identifier: MIT OR Apache-2.0
# Recursive functions: factorial, Fibonacci, Ackermann and mutual recursion.


def fact(n):
    return 1 if n <= 1 else n * fact(n - 1)


def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


def ack(m, n):
    if m == 0:
        return n + 1
    if n == 0:
        return ack(m - 1, 1)
    return ack(m - 1, ack(m, n - 1))


def is_even(n):
    return True if n == 0 else is_odd(n - 1)


def is_odd(n):
    return False if n == 0 else is_even(n - 1)


for i in range(0, 30, 5):
    print(i, fact(i))
print([fib(i) for i in range(15)])
print(ack(2, 3), ack(3, 3))
print(is_even(10), is_odd(7), is_even(7))
