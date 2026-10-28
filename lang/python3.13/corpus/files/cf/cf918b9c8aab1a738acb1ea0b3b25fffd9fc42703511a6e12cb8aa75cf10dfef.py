# SPDX-License-Identifier: MIT OR Apache-2.0
# User exception classes: the first matching except clause wins, str and
# repr of exceptions, args, and an uncaught user exception at the end.


class AppError(Exception):
    pass


class NotFound(AppError):
    def __init__(self, key):
        super().__init__(f"missing {key}")
        self.key = key


class Invalid(AppError, ValueError):
    pass


for exc in [NotFound("k"), Invalid("bad"), AppError(), ValueError("v")]:
    try:
        raise exc
    except NotFound as e:
        print("NotFound", e, e.key, e.args)
    except ValueError as e:
        print("ValueError branch", type(e).__name__, repr(e))
    except AppError as e:
        print("AppError", repr(e), str(e) == "")
print(Invalid.__mro__)
print(str(KeyError("k")), str(KeyError("a", "b")), repr(OSError(2, "No such file")), str(Exception()))
raise NotFound("final")
