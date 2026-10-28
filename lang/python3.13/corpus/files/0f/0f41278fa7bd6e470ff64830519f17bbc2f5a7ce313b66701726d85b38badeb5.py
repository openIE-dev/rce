# SPDX-License-Identifier: MIT OR Apache-2.0
# Scope rules: a name assigned anywhere in a function is local throughout
# it, global and nonlocal declarations, and class bodies that do not
# enclose their methods.
x = 10


def reads():
    return x


def shadows():
    try:
        print(x)
    except UnboundLocalError as e:
        print("UnboundLocalError", e)
    x = 1
    return x


def writes():
    global x
    x += 5


print(reads(), shadows(), x)
writes()
print(x)


class C:
    y = 3
    doubled = [y * 2 for _ in range(2)] if False else None

    def m(self):
        try:
            return y
        except NameError as e:
            return f"NameError {e}"


print(C().m(), C.y, C.doubled)


def outer():
    v = "outer"

    def mid():
        def inner():
            nonlocal v
            v = "changed"

        inner()

    mid()
    return v


print(outer())
