// SPDX-License-Identifier: MIT OR Apache-2.0
// Case mapping and normalization follow the Unicode data the engine
// carries: full mappings may change the length.
for (const w of ["straße", "İstanbul", "ǅemal", "ΣΊΣΥΦΟΣ", "ﬁne", "Ⅻ", "ǈ", "ŉ"]) {
  console.log(w, w.toUpperCase(), w.toLowerCase(), w.length, w.toUpperCase().length, w.toLowerCase().length);
}
console.log(JSON.stringify(" \t\n\u00a0\u2028\ufeff\u3000x\u200b".trim()), "\u180e".trim().length, "ab".padStart(5, "😀"));
console.log(["NFC", "NFD", "NFKC", "NFKD"].map((f) => "é①ﬁ".normalize(f).length).join(","), "Å".normalize("NFD").length);
console.log("İ".toLowerCase().length, "ı".toUpperCase(), "ς".toUpperCase(), "ΑΣ".toLowerCase(), "a\u0301".localeCompare === undefined);
