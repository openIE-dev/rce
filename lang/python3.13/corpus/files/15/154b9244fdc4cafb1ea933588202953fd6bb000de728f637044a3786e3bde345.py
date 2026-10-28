# SPDX-License-Identifier: MIT OR Apache-2.0
# Exception chaining (raise from, implicit context), return in finally
# overriding a raise, and exception groups.


def override():
    try:
        raise ValueError("lost")
    finally:
        return "finally wins"


print(override())
try:
    try:
        {}["k"]
    except KeyError as e:
        raise RuntimeError("wrapped") from e
except RuntimeError as e:
    print(repr(e), repr(e.__cause__), e.__suppress_context__)
try:
    try:
        1 / 0
    except ZeroDivisionError:
        int("x")
except ValueError as e:
    print(repr(e), type(e.__context__).__name__, e.__cause__)
try:
    raise ValueError("no context") from None
except ValueError as e:
    print(e.__cause__, e.__suppress_context__)
try:
    raise ExceptionGroup("many", [ValueError("a"), TypeError("b"), ValueError("c")])
except* ValueError as eg:
    print("values", [str(x) for x in eg.exceptions])
except* TypeError as eg:
    print("types", [str(x) for x in eg.exceptions])
order = []
try:
    try:
        order.append("try")
        raise KeyError("x")
    finally:
        order.append("finally")
except KeyError:
    order.append("except")
print(order)
