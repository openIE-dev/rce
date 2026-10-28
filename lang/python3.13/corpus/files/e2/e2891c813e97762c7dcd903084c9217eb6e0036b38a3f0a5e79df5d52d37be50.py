# SPDX-License-Identifier: MIT OR Apache-2.0
# Boolean operators return one of their operands and evaluate the right one
# only when needed.
calls = []


def t(name):
    calls.append(name)
    return True


def f(name):
    calls.append(name)
    return False


print(True and False, True or False, not True, not None)
r1 = f("a") and t("b")
r2 = t("c") or f("d")
r3 = t("e") and f("f") or t("g")
r4 = not (f("h") or f("i"))
print(r1, r2, r3, r4)
print(calls)
print(0 or "default", "" or [] or None, 3 and 4, 0 and 1 / 0, [] or {}, "x" and "y")
print(bool(0), bool(0.0), bool(""), bool([]), bool({}), bool(None), bool("0"), bool([0]), bool(-1))
x = 5
print(1 < x < 10, 1 < x > 3, x == 5 != 6, not 1 < x)
