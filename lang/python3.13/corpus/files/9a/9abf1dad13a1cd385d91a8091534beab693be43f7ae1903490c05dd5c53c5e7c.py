# SPDX-License-Identifier: MIT OR Apache-2.0
# Default arguments, *args and **kwargs, keyword-only and positional-only
# parameters, several results as a tuple, and functions as values.


def divmod2(a, b):
    return a // b, a % b


def greet(name, greeting="hello", *, punct="!"):
    return f"{greeting}, {name}{punct}"


def total(*nums, start=0):
    return start + sum(nums)


def info(pos, /, *args, **kwargs):
    return f"{pos} {args} {sorted(kwargs.items())}"


def apply(f, xs):
    return [f(x) for x in xs]


print(divmod2(17, 5))
q, r = divmod2(-17, 5)
print(q, r)
print(greet("ann"), greet("bob", "hi"), greet(greeting="yo", name="cid"), greet("dee", punct="?"))
print(total(), total(1, 2, 3), total(*[4, 5], start=10))
print(info(1), info(1, 2, 3, b=2, a=1))
print(apply(lambda x: x * x, [1, 2, 3]), apply(abs, [-1, 2]), apply(str.upper, ["a"]))
try:
    greet("x", "y", "z")
except TypeError as e:
    print("TypeError", e)
try:
    info(pos=1)
except TypeError as e:
    print("TypeError", e)
print(greet.__name__, greet.__defaults__, greet.__kwdefaults__)
