# SPDX-License-Identifier: MIT OR Apache-2.0
# repr of strings and bytes: the choice of quotes, escapes for characters
# that are not printable, and ascii, which escapes everything past ASCII.
for s in ["plain", "it's", 'say "hi"', "both ' and \"", "tab\there", "nl\n", "\x00\x1f\x7f", "é", "\u200b", "\u2028",
          "😀", "\ud800", "\\", "\r\n", "\x85", "\xa0"]:
    print(repr(s), ascii(s), len(s))
for b in [b"bytes", b"it's", b"\x00\xff", b'"', bytes(range(0x7c, 0x84))]:
    print(repr(b), list(b)[:4])
print(repr("a" * 3), str(["é", "\n"]), repr(("x",)), repr({"k": "v\t"}), repr(bytearray(b"\x01")))
