// SPDX-License-Identifier: MIT OR Apache-2.0
// sort with a comparator, which ECMAScript 2019 and later require to be
// stable, and sorting by several keys.
const xs = [5, 2, 9, 1, 5, 6, -3, 0];
console.log([...xs].sort((a, b) => a - b), [...xs].sort((a, b) => b - a), xs);
const ws = ["pear", "Apple", "fig", "banana", "apple"];
console.log([...ws].sort(), [...ws].sort((a, b) => (a.toLowerCase() < b.toLowerCase() ? -1 : a.toLowerCase() > b.toLowerCase() ? 1 : 0)));
console.log([...ws].sort((a, b) => a.length - b.length));
const people = [["Ann", 30], ["Bob", 25], ["Cid", 30], ["Dee", 25], ["Eve", 35]];
console.log(JSON.stringify([...people].sort((a, b) => a[1] - b[1])));
console.log(JSON.stringify([...people].sort((a, b) => b[1] - a[1])));
const recs = [[2, "b"], [1, "z"], [2, "a"], [1, "a"]];
console.log(JSON.stringify(recs.sort((a, b) => a[0] - b[0] || (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0))));
console.log([3.5, 1, 2.25, -1, 0].sort((a, b) => a - b), ["b", "a", "c"].sort().reverse());
