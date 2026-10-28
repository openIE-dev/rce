# SPDX-License-Identifier: MIT OR Apache-2.0
# Generators run lazily: yield, send, return values carried by
# StopIteration, yield from, and closing.


def counter(n):
    i = 0
    while i < n:
        received = yield i
        i += 1 if received is None else received
    return "done"


g = counter(10)
print(next(g), next(g), g.send(3), next(g))
g2 = counter(1)
next(g2)
try:
    next(g2)
except StopIteration as e:
    print("StopIteration", e.value)


def outer():
    result = yield from counter(2)
    yield result


print(list(outer()))
log = []


def lazy():
    for i in range(3):
        log.append(f"make {i}")
        yield i


it = lazy()
log.append("created")
first = next(it)
log.append(f"got {first}")
print(log, list(it), log[-1])


def closer():
    try:
        yield 1
        yield 2
    finally:
        print("cleanup")


c = closer()
print(next(c))
c.close()
print(sum(x for x in range(5)), [y for y in (x * x for x in range(4)) if y])
