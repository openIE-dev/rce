# SPDX-License-Identifier: MIT OR Apache-2.0
# Closures capture variables, not values: nonlocal rebinds a captured
# variable, and lambdas made in a loop all see the loop variable's last
# value unless it is bound as a default argument.


def counter():
    n = 0

    def inc():
        nonlocal n
        n += 1
        return n

    return inc


c1, c2 = counter(), counter()
print(c1(), c1(), c1(), c2())


def adder(base):
    return lambda x: base + x


add5 = adder(5)
print(add5(1), add5(10))
late = [lambda: i * 10 for i in range(3)]
print([f() for f in late])
bound = [lambda i=i: i * 10 for i in range(3)]
print([f() for f in bound])


def fib_gen():
    a, b = 0, 1

    def step():
        nonlocal a, b
        r = a
        a, b = b, a + b
        return r

    return step


step = fib_gen()
print([step() for _ in range(8)])
x = "global"


def outer():
    x = "enclosing"

    def inner():
        return x

    return inner()


print(outer(), x)
