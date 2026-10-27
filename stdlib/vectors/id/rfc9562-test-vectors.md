# Test Vectors {#test_vectors}

Both UUIDv1 and UUIDv6 test vectors utilize the same 60 bit timestamp: 0x1EC9414C232AB00
(138648505420000000) Tuesday, February 22, 2022 2:22:22.000000 PM GMT-05:00

Both UUIDv1 and UUIDv6 utilize the same values in clock_seq,
and node. All of which have been generated with random data.
For the randomized node, the least significant bit of the first octet is set to a value of 1 as per {{unidentifiable}}. 
Thus the starting value 0x9E6BDECED846 was changed to 0x9F6BDECED846.

The pseudocode used for converting from a 64 bit Unix timestamp to a 100ns Gregorian timestamp value
has been left in the document for reference purposes.

~~~~ code
# Gregorian to Unix Offset:
# The number of 100-ns intervals between the
# UUID epoch 1582-10-15 00:00:00
# and the Unix epoch 1970-01-01 00:00:00
# Greg_Unix_offset = 0x01b21dd213814000 or 122192928000000000

# Unix 64 bit Nanosecond Timestamp:
# Unix NS: Tuesday, February 22, 2022 2:22:22 PM GMT-05:00
# Unix_64_bit_ns = 0x16D6320C3D4DCC00 or 1645557742000000000

# Unix Nanosecond precision to Gregorian 100-nanosecond intervals
# Greg_100_ns = (Unix_64_bit_ns/100)+Greg_Unix_offset

# Work:
# Greg_100_ns = (1645557742000000000/100)+122192928000000000
# Unix_64_bit_ns = (138648505420000000-122192928000000000)*100

# Final:
# Greg_100_ns = 0x1EC9414C232AB00 or 138648505420000000
~~~~
{: title='Test Vector Timestamp Pseudo-code'}

## Example of a UUIDv1 Value {#uuidv1_example}

~~~~
-------------------------------------------
field      bits value
-------------------------------------------
time_low   32   0xC232AB00
time_mid   16   0x9414
ver         4   0x1
time_high  12   0x1EC
var         2   0b10
clock_seq  14   0b11, 0x3C8
node       48   0x9F6BDECED846
-------------------------------------------
total      128
-------------------------------------------
final: C232AB00-9414-11EC-B3C8-9F6BDECED846
~~~~
{: title='UUIDv1 Example Test Vector'}

## Example of a UUIDv3 Value {#uuidv3_example}
The MD5 computation from is detailed in {{v3md5}} using the DNS Namespace ID value and the Name "www.example.com".
while the field mapping and all values are illustrated in {{v3fields}}.
Finally to further illustrate the bit swapping for version and variant see {{v3vervar}}.

~~~~
Namespace (DNS):  6ba7b810-9dad-11d1-80b4-00c04fd430c8
Name:             www.example.com
------------------------------------------------------
MD5:              5df418813aed051548a72f4a814cf09e
~~~~
{: id='v3md5' title='UUIDv3 Example MD5'}

~~~~
-------------------------------------------
field     bits value
-------------------------------------------
md5_high  48   0x5df418813aed
ver        4   0x3
md5_mid   12   0x515
var        2   0b10
md5_low   62   0b00, 0x8a72f4a814cf09e
-------------------------------------------
total     128
-------------------------------------------
final: 5df41881-3aed-3515-88a7-2f4a814cf09e
~~~~
{: id='v3fields' title='UUIDv3 Example Test Vector'}

~~~~
MD5 hex and dash:      5df41881-3aed-0515-48a7-2f4a814cf09e
Ver and Var Overwrite: xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx
Final:                 5df41881-3aed-3515-88a7-2f4a814cf09e
~~~~
{: id='v3vervar' title='UUIDv3 Example Ver Var bit swaps'}

## Example of a UUIDv4 Value {#uuidv4_example}
This UUIDv4 example was created by generating 16 bytes
of random data resulting in the hexadecimal value of
919108F752D133205BACF847DB4148A8. This is then used to
fill out the fields as shown in {{v4fields}}.

Finally to further illustrate the bit swapping for version and variant see {{v4vervar}}.

~~~~
-------------------------------------------
field     bits value
-------------------------------------------
random_a  48   0x919108f752d1
ver        4   0x4
random_b  12   0x320
var        2   0b10
random_c  62   0b01, 0xbacf847db4148a8
-------------------------------------------
total     128
-------------------------------------------
final: 919108f7-52d1-4320-9bac-f847db4148a8
~~~~
{: id='v4fields' title='UUIDv4 Example Test Vector'}

~~~~
Random hex:            919108f752d133205bacf847db4148a8
Random hex and dash:   919108f7-52d1-3320-5bac-f847db4148a8
Ver and Var Overwrite: xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx
Final:                 919108f7-52d1-4320-9bac-f847db4148a8
~~~~
{: id='v4vervar' title='UUIDv4 Example Ver/Var bit swaps '}


## Example of a UUIDv5 Value {#uuidv5_example}
The SHA-1 computation from is detailed in {{v5sha1}} using the DNS Namespace ID value and the Name "www.example.com".
while the field mapping and all values are illustrated in {{v5fields}}.
Finally to further illustrate the bit swapping for version and variant and the unused/discarded part of the SHA-1 value see {{v5vervar}}.

~~~~
Namespace (DNS):  6ba7b810-9dad-11d1-80b4-00c04fd430c8
Name:             www.example.com
----------------------------------------------------------
SHA-1:            2ed6657de927468b55e12665a8aea6a22dee3e35
~~~~
{: id='v5sha1' title='UUIDv5 Example SHA-1'}

~~~~
-------------------------------------------
field      bits value
-------------------------------------------
sha1_high  48   0x2ed6657de927
ver         4   0x5
sha1_mid   12   0x68b
var         2   0b10
sha1_low   62   0b01, 0x5e12665a8aea6a2
-------------------------------------------
total      128
-------------------------------------------
final: 2ed6657d-e927-568b-95e1-2665a8aea6a2
~~~~
{: id='v5fields' title='UUIDv5 Example Test Vector'}

~~~~
SHA-1 hex and dash:    2ed6657d-e927-468b-55e1-2665a8aea6a2-2dee3e35
Ver and Var Overwrite: xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx
Final:                 2ed6657d-e927-568b-95e1-2665a8aea6a2
Discarded:                                                 -2dee3e35
~~~~
{: id='v5vervar' title='UUIDv5 Example Ver/Var bit swaps and discarded SHA-1 segment'}

## Example of a UUIDv6 Value {#uuidv6_example}

~~~~
-------------------------------------------
field       bits value
-------------------------------------------
time_high   32   0x1EC9414C
time_mid    16   0x232A
ver          4   0x6
time_high   12   0xB00
var          2   0b10
clock_seq   14   0b11, 0x3C8
node        48   0x9F6BDECED846
-------------------------------------------
total       128
-------------------------------------------
final: 1EC9414C-232A-6B00-B3C8-9F6BDECED846
~~~~
{: title='UUIDv6 Example Test Vector'}


## Example of a UUIDv7 Value {#uuidv7_example}

This example UUIDv7 test vector utilizes a well-known Unix epoch timestamp with
millisecond precision to fill the first 48 bits.

rand_a and rand_b are filled with random data.

The timestamp is Tuesday, February 22, 2022 2:22:22.00 PM GMT-05:00 represented
as 0x017F22E279B0 or 1645557742000


~~~~
-------------------------------------------
field       bits value
-------------------------------------------
unix_ts_ms  48   0x017F22E279B0
ver          4   0x7
rand_a      12   0xCC3
var          2   0b10
rand_b      62   0b01, 0x8C4DC0C0C07398F
-------------------------------------------
total       128
-------------------------------------------
final: 017F22E2-79B0-7CC3-98C4-DC0C0C07398F
~~~~
{: title='UUIDv7 Example Test Vector'}

# Illustrative Examples {#ill_examples}
The following sections contain illustrative examples which serve to show how one may use UUIDv8 {{uuidv8}} for custom and/or experimental application based logic.
The examples below have not been through the same rigorous testing, prototyping, and feedback loop that other algorithms in this document have undergone.
The authors encouraged implementors to create your own UUIDv8 algorithm rather than use the items defined in this section.

## Example of a UUIDv8 Value (time-based) {#uuidv8_example}
This example UUIDv8 test vector utilizes a well-known 64 bit Unix epoch timestamp with
10ns precision, truncated to the least-significant, right-most, bits
to fill the first 60 bits of custom_a and custom_b while setting the version bits between these two segments to the version value of 8.

The variant bits are set and the final segment, custom_c, is filled with random data.

Timestamp is Tuesday, February 22, 2022 2:22:22.000000 PM GMT-05:00 represented
as 0x2489E9AD2EE2E00 or 164555774200000000 (10ns-steps).

~~~~
-------------------------------------------
field     bits value
-------------------------------------------
custom_a  48   0x2489E9AD2EE2
ver        4   0x8
custom_b  12   0xE00
var        2   0b10
custom_c  62   0b00, 0xEC932D5F69181C0
-------------------------------------------
total     128
-------------------------------------------
final: 2489E9AD-2EE2-8E00-8EC9-32D5F69181C0
~~~~
{: title='UUIDv8 Example Time-based Illustrative Example'}

## Example of a UUIDv8 Value (name-based) {#uuidv8_example_name}
As per {{uuidv5}} name-based UUIDs that desire to use modern hashing algorithms MUST be created within the UUIDv8 space.
These MAY leverage newer hashing algorithms such as SHA-256 or SHA-512 defined by {{FIPS180-4}}, SHA-3 or SHAKE defined by {{FIPS202}}, or even algorithms that have not been defined yet.

A SHA-256 version of {{uuidv5_example}} is detailed in {{v8sha256}} as an illustrative example detailing how this can be achieved.
The creation of the name-based UUIDv8 value in this section follows the same logic defined in {{uuidv5}} with the difference being SHA-256 in place of SHA-1.

The field mapping and all values are illustrated in {{v8fieldssha256}}.
Finally to further illustrate the bit swapping for version and variant and the unused/discarded part of the SHA-256 value see {{v8vervar}}.
An important note for secure hashing algorithms that produce outputs of an arbitrary size, such as those found in SHAKE, the output hash MUST be 128 bits or larger.


~~~~
Namespace (DNS):       6ba7b810-9dad-11d1-80b4-00c04fd430c8
Name:                  www.example.com
----------------------------------------------------------------
SHA-256:
5c146b143c524afd938a375d0df1fbf6fe12a66b645f72f6158759387e51f3c8
~~~~
{: id='v8sha256' title='UUIDv8 Example SHA256'}

~~~~
-------------------------------------------
field     bits value
-------------------------------------------
custom_a  48   0x5c146b143c52
ver        4   0x8
custom_b  12   0xafd
var        2   0b10
custom_c  62   0b00, 0x38a375d0df1fbf6
-------------------------------------------
total     128
-------------------------------------------
final: 5c146b14-3c52-8afd-938a-375d0df1fbf6
~~~~
{: id='v8fieldssha256' title='UUIDv8 Example Name-Based SHA-256 Illustrative Example'}

~~~~
A: 5c146b14-3c52-4afd-938a-375d0df1fbf6-fe12a66b645f72f6158759387e51f3c8
B: xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx
C: 5c146b14-3c52-8afd-938a-375d0df1fbf6
D:                                     -fe12a66b645f72f6158759387e51f3c8
~~~~
{: id='v8vervar' title='UUIDv8 Example Ver/Var bit swaps and discarded SHA-256 segment'}

Examining {{v8vervar}}:

{: spacing="compact"}

- Line A details the full SHA-256 as a hexadecimal value with the dashes inserted.
- Line B details the version and variant hexadecimal positions which must be overwritten.
- Line C details the final value after the ver/var have been overwritten.
- Line D details the discarded, leftover values from the original SHA-256 computation.
