// SPDX-License-Identifier: MIT OR Apache-2.0
// How console.log shows values: strings as they are at the top level,
// everything else as Node's inspection prints it.
console.log(42, -7, 3.5, true, null, undefined, "text", 10n);
console.log([1, 2, 3], [], {}, { a: 1, b: "x" }, [["n"], { k: [1, { m: [] }] }]);
console.log(["a", "it's", 'say "hi"', "nl\n"], [null, undefined, true]);
console.log([1.0, 0.1, 1e100, -0, Infinity, NaN], -0, 0);
console.log(new Map([[1, "a"], ["b", [2]]]), new Set([3, "x"]));
console.log(String(null), String(undefined), String([1, [2, 3]]), String({}), `${[]}`, "" + 1 + 2, 1 + 2 + "");
console.log("a", "b", 1, 2);
console.log();
console.log([function named() {}, () => 1, class K {}]);
console.log(Symbol("s"), [Symbol.iterator], typeof Symbol());
