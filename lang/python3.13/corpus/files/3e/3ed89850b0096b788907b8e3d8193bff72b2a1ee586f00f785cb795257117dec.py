# SPDX-License-Identifier: MIT OR Apache-2.0
# The CRC-32 of IEEE 802.3 computed with a table built at run time, checked
# against zlib.crc32 and binascii.crc32.
import binascii
import zlib

table = []
for i in range(256):
    c = i
    for _ in range(8):
        c = 0xEDB88320 ^ (c >> 1) if c & 1 else c >> 1
    table.append(c)
for s in [b"", b"a", b"123456789", b"The quick brown fox jumps over the lazy dog"]:
    c = 0xFFFFFFFF
    for b in s:
        c = table[(c ^ b) & 0xFF] ^ (c >> 8)
    c ^= 0xFFFFFFFF
    print(f"{c:08x}", c == zlib.crc32(s), c == binascii.crc32(s))
