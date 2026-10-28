# SPDX-License-Identifier: MIT OR Apache-2.0
# Base64 encoding written with shifts and masks, checked against the base64
# module and decoded back.
import base64

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def encode(b):
    out = []
    for i in range(0, len(b), 3):
        chunk = b[i:i + 3]
        n = int.from_bytes(chunk + b"\x00" * (3 - len(chunk)), "big")
        for j in range(4):
            out.append(ALPHABET[(n >> (18 - 6 * j)) & 63] if j <= len(chunk) else "=")
    return "".join(out)


for s in [b"", b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar", "héllo wörld".encode(), b"\x00\xff\xfe"]:
    e = encode(s)
    print(repr(s), e, e == base64.b64encode(s).decode(), base64.b64decode(e) == s)
