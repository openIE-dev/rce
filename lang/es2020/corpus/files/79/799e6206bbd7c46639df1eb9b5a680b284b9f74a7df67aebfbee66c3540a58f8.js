// SPDX-License-Identifier: MIT OR Apache-2.0
// BigInt: exact integers, division truncating toward zero, no implicit
// mixing with Number, and conversions that throw.
console.log(2n ** 64n, -(2n ** 63n), 2n ** 100n, -7n / 2n, -7n % 2n, 7n / -2n, 7n % -2n, 5n / 3n);
console.log(BigInt(Number.MAX_SAFE_INTEGER) + 2n, BigInt("0x1f"), BigInt("  42  "), BigInt(true), BigInt.asIntN(8, 255n), BigInt.asUintN(8, -1n));
console.log(1n == 1, 1n === 1, 2n > 1, 1n < 1.5, typeof 1n, 0n ? "truthy" : "falsy", String(10n), (255n).toString(16), -(-5n));
for (const f of [() => 1n + 1, () => BigInt(1.5), () => BigInt("1.5"), () => JSON.stringify(1n), () => Math.max(1n), () => 1n >>> 0n, () => +1n]) {
  try {
    console.log(f());
  } catch (e) {
    console.log(e.name, e.message);
  }
}
console.log(Number(2n ** 64n), Number(2n ** 53n + 1n), parseInt("12n"), [3n, 1n, 2n].sort().join(","), [10n, 9n].sort((a, b) => (a < b ? -1 : 1)));
