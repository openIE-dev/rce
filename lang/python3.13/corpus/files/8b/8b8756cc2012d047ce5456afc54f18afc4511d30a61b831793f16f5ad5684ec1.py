# SPDX-License-Identifier: MIT OR Apache-2.0
# try, except, else and finally, the order in which they run, and raising.


def trial(x):
    out = []
    try:
        out.append("try")
        if x == 0:
            raise ValueError("zero")
        if x == 1:
            return out + ["returned"]
        out.append(str(10 // (x - 2)))
    except ValueError as e:
        out.append(f"ValueError({e})")
    except ZeroDivisionError:
        out.append("ZeroDivisionError")
    else:
        out.append("else")
    finally:
        out.append("finally")
    return out


for x in range(4):
    print(x, trial(x))
try:
    int("abc")
except (TypeError, ValueError) as e:
    print(type(e).__name__, e)
try:
    [][1]
except LookupError as e:
    print(type(e).__name__, e.args)
try:
    raise KeyError("k")
except Exception as e:
    print(repr(e), str(e), e.args)
try:
    try:
        1 / 0
    except ZeroDivisionError:
        raise
except ArithmeticError as e:
    print("re-raised", type(e).__name__, e)
