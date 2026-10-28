# SPDX-License-Identifier: MIT OR Apache-2.0
# f-strings, str.format and %-formatting, with widths, alignment, signs and
# conversions.
n, s, x = 42, "go", 3.14159
print(f"{n}|{n:5d}|{n:<5d}|{n:05d}|{n:+d}|{n:x}|{n:X}|{n:o}|{n:b}|{n:#x}|{n:#o}|{n:#b}|{n:c}")
print(f"{s}|{s:>10}|{s:<10}|{s:^10}|{s:*^10}|{'golang':.3}|{s!r}|{s!a}|{'é'!a}")
print(f"{x:.2f}|{x:10.3f}|{x:e}|{x:.0f}|{-x:+.1f}|{x:=+10.2f}|{1234567:,}|{1234567:_d}|{0.25:.1%}")
print("{0} {1} {0}".format("a", "b"), "{name}={value}".format(name="k", value=1), "{:>{w}}".format("r", w=4))
print("%d %s %r %5.1f %-5s| %x %o %e %c %%" % (7, "s", "s", 2.25, "ab", 255, 8, 12345.678, 65))
print("%(a)s and %(b)s" % {"a": 1, "b": 2})
width, prec = 8, 3
print(f"{x:{width}.{prec}f}|{n=}|{n * 2 = }|{s=!r:>6}")
print(f"{'nested ' + f'{n}'}", f"{{literal}}", f"{[1, 2][0]}", f"{ {'k': 1}['k'] }")
print(format(True, "d"), format(True), f"{None}", f"{None!r:>6}", format(-0.0, ".1f"), format(1e16, ","))
