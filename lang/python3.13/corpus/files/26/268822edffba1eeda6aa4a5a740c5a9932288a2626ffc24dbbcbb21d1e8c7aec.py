# SPDX-License-Identifier: MIT OR Apache-2.0
# The text that repr, str and format give float values.
vals = [0.0, 1.0, -1.0, 0.5, 0.1, 1 / 3, 100.0, 123456.0, 1234567.0, 1e6, 1e15, 1e16, 1e20, 1e21, 1e-4, 1e-5, 12345.678, 2.5e-8]
for v in vals:
    print(f"{v!r}|{v}|{v:g}|{v:e}|{v:f}|{v:.2f}|{v:8.3f}|{v:<8.1f}|{v:G}|{v:%}")
print(f"{3.14159:5.1f}|{3.14159:05.1f}|{3.14159:+.1f}|{3.14159: .1f}|{1234567.891:,.2f}|{1234567.891:_.1f}")
print(format(1e16, ".0f"), format(0.1, ".20f"), format(2 ** 0.5, ".30g"))
print("%s %r %.3f %e %g %5.1f%%" % (0.1, 0.1, 0.1, 0.1, 0.1, 99.44))
