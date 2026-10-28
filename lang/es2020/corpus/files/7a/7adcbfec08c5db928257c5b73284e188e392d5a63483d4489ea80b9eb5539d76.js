// SPDX-License-Identifier: MIT OR Apache-2.0
// Word frequencies of a fixed text, sorted by count and then by word.
const TEXT = `It was the best of times, it was the worst of times, it was the age of
wisdom, it was the age of foolishness, it was the epoch of belief, it was the
epoch of incredulity, it was the season of Light, it was the season of Darkness`;
const words = TEXT.toLowerCase().match(/[a-z]+/g);
const counts = new Map();
for (const w of words) counts.set(w, (counts.get(w) || 0) + 1);
const list = [...counts].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
for (const [w, c] of list) console.log(c, w);
console.log(words.length, counts.size, [...counts.keys()].slice(0, 5).join(","));
