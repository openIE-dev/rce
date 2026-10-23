# Hardware: bits

Programs that work at the level of bits, where every result must be
exact.

| Program | What it shows |
|---|---|
| `bits.rce` | population count, extracting a bit field, and CRC-32 with a lookup table computed by a `comptime` function while the program is checked |

The CRC-32 check uses the standard test input `"123456789"`, whose CRC-32
is `0xCBF43926`.

Limits in draft 0.5: there are no bit-width signal types and no hardware
description emitter.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
