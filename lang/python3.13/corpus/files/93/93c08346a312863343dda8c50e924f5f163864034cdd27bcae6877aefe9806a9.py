# SPDX-License-Identifier: MIT OR Apache-2.0
# Case mapping and classification follow the Unicode data CPython carries:
# full mappings may change the length, and digits, decimals and numerics
# differ.
import unicodedata

for w in ["straße", "İstanbul", "ǅemal", "ΣΊΣΥΦΟΣ", "ﬁne", "Ⅻ", "ǈ", "ŉ"]:
    print(w, w.upper(), w.lower(), w.casefold(), w.title(), len(w), len(w.upper()), len(w.lower()))
for c in "a5٣²½Ⅻ\u00a0\u2028_日":
    print(f"U+{ord(c):04X}", c.isalpha(), c.isdecimal(), c.isdigit(), c.isnumeric(), c.isspace(), c.isidentifier(), c.isprintable())
print("straße".casefold() == "STRASSE".casefold(), "Σ".lower(), "ΟΔΟΣ".lower(), "ς".upper())
print(unicodedata.unidata_version, unicodedata.name("é"), unicodedata.category("²"), unicodedata.numeric("½"))
print([unicodedata.normalize(f, "é") for f in ("NFC", "NFD")], len(unicodedata.normalize("NFD", "é")), unicodedata.normalize("NFKC", "ﬁ²"))
print("e\u0301" == "é", unicodedata.normalize("NFC", "e\u0301") == "é")
